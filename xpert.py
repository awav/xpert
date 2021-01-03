# Copyright 2020 by Artem Artemev, @awav.
# All rights reserved.

import click
import hashlib
import asyncio
import glob
import itertools
import os
import re
import time

# from collections.abc import Sequence
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    Generator,
    Iterator,
    List,
    Mapping,
    MutableSet,
    Optional,
    Sequence,
    Union,
    TypeVar,
)

import toml
from termcolor import colored


@dataclass(frozen=True)
class Unit:
    cmd: str
    uid: str


def _cpu_count() -> int:
    count = os.cpu_count()
    return count if count else 1


@dataclass(frozen=True)
class Flags:
    restart: bool = False
    num_proc: int = field(default_factory=_cpu_count)
    gpu_indices: Optional[List[str]] = None

    name: str = field(init=False, default="flags")

    def __post_init__(self):
        if self.gpu_indices is not None and self.num_proc > len(self.gpu_indices):
            raise ValueError(
                "`flags.num_proc` should be smaller than then number of available GPUs"
            )


TomlLiteral = Union[int, float, str]
TomlSequence = List[TomlLiteral]
TomlValue = Union[TomlLiteral, TomlSequence]
TomlDict = Dict[str, Any]


def _is_option(opt: Dict) -> bool:
    if not isinstance(opt, dict):
        return False
    keys = set(opt.keys())
    return keys.issubset(["value", "type", "xprod"])


def _process_unit_option(opt: Dict) -> TomlValue:
    value = opt["value"]
    value_type = opt.get("type", None)
    value_xprod = opt.get("xprod", True)

    if value_type.lower() == "path":
        value = [str(Path(v).resolve()) for v in glob.glob(value)]

    # TODO(awav): do not ignore `value_xprod`
    return value


@dataclass(frozen=True)
class UnitSetup:
    exp: Dict[str, TomlValue]
    cmd: Optional[str] = None
    uid: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.exp, dict):
            raise ValueError("`exp` must be a dictionary")

        for key, value in self.exp.items():
            if not isinstance(value, dict):
                continue

            if not _is_option(value):
                raise ValueError(f"`exp.{key}` has invalid attributes {value}")

            self.exp[key] = _process_unit_option(value)

    def gen_experiments(
        self, global_cmd: Union[str, None], global_uid: Union[str, None]
    ) -> Generator:
        class missingdict(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        uid = self.uid.format_map(missingdict(uid=global_uid)) if self.uid else global_uid
        cmd = self.cmd.format_map(missingdict(cmd=global_cmd)) if self.cmd else global_cmd

        if cmd is None:
            raise ValueError("Either `global_cmd` or local experiment `cmd` should be set")

        if uid is None:
            raise ValueError("Either `global_uid` or local experiment `uid` should be set")

        def is_xprod(v: Any) -> bool:
            if not isinstance(v, str) and isinstance(v, Sequence):
                return True
            return False

        opts = self.exp.items()
        linear = {k: v for k, v in opts if not is_xprod(v)}
        xprod_dict = {k: v for k, v in opts if is_xprod(v)}
        xprod_dict_keys = xprod_dict.keys()
        xprod_dict_values = xprod_dict.values()
        for xprod_values in itertools.product(*xprod_dict_values):
            xprod = dict(zip(xprod_dict_keys, xprod_values))
            uid_exp = uid.format(**linear, **xprod)
            cmd_exp = cmd.format(uid=uid, **linear, **xprod)
            yield Unit(cmd_exp, uid_exp)


@dataclass(frozen=True)
class Config:
    toml_config: InitVar[TomlDict]

    global_cmd: str = field(init=False)
    global_uid: str = field(init=False)
    flags: Flags = field(init=False)
    experiment_settings: List[UnitSetup] = field(init=False)

    def __post_init__(self, toml_config: TomlDict):
        flags_args = toml_config[Flags.name]
        flags = Flags(**flags_args)
        super().__setattr__("flags", flags)

        cmd = toml_config.get("cmd", None)
        super().__setattr__("global_cmd", cmd)

        uid = toml_config.get("uid", None)
        super().__setattr__("global_uid", uid)

        experiments = toml_config["exp"]
        if not isinstance(experiments, list):
            experiments = [experiments]

        experiment_settings = []
        for e in experiments:
            e_copy = e.copy()
            local_cmd = e_copy.pop("cmd", None)
            local_uid = e_copy.pop("uid", None)
            u = UnitSetup(cmd=local_cmd, uid=local_uid, exp=e_copy)
            experiment_settings.append(u)

        super().__setattr__("experiment_settings", experiment_settings)

    def gen_experiments(self) -> Generator:
        cmd = self.global_cmd
        uid = self.global_uid
        for setting in self.experiment_settings:
            yield from setting.gen_experiments(cmd, uid)


def _quoted_split(string: str):
    r"""
    StackOverflow solution for splitting string with spaces ignoring the content
    inside quotes ' and ".
    https://stackoverflow.com/a/51560564/7788672
    """

    def strip_quotes(s):
        if s and (s[0] == '"' or s[0] == "'") and s[0] == s[-1]:
            return s[1:-1]
        return s

    return [
        strip_quotes(p).replace('\\"', '"').replace("\\'", "'")
        for p in re.findall(r'"(?:\\.|[^"])*"|\'(?:\\.|[^\'])*\'|[^\s]+', string)
    ]


@dataclass
class RunContext:
    config: InitVar[Config]

    lock: asyncio.Lock = field(init=False)
    restart: bool = field(init=False)
    gpu_indices: Optional[FrozenSet[str]] = field(init=False, default=None)
    gpu_indices_shared: Optional[MutableSet[str]] = field(init=False, default=None)

    def __post_init__(self, config: Config):
        self.lock = asyncio.Lock()
        flags = config.flags
        self.restart = flags.restart
        indices = flags.gpu_indices
        if indices is not None:
            self.gpu_indices = frozenset(deepcopy(indices))
            self.gpu_indices_shared = set(deepcopy(indices))

    def program_and_args(self, cmd: str) -> List[str]:
        return _quoted_split(cmd)

    def output_log_file(self, uid: str) -> Path:
        timestamp = int(time.time())
        return self._filepath(f"stdout.{timestamp}.log", uid)

    def error_log_file(self, uid: str) -> Path:
        timestamp = int(time.time())
        return self._filepath(f"stderr.{timestamp}.log", uid)

    def completed_file(self, uid: str) -> Path:
        return self._filepath("completed.flag", uid)

    def failed_file(self, uid: str) -> Path:
        return self._filepath("failed.flag", uid)

    def _filepath(self, status: str, uid: str) -> Path:
        base = Path(uid)
        if not base.exists():
            base.mkdir(parents=True)
        return Path(base, status).resolve()

    def is_task_completed(self, uid: str) -> bool:
        if self.restart:
            return False
        return self.completed_file(uid).exists()

    async def pop_gpu_indices(self) -> str:
        if self.gpu_indices is not None:
            async with self.lock:
                if len(self.gpu_indices_shared) == 0:
                    raise RuntimeError(
                        f"GPU indices set is empty." f"Original set: {self.gpu_indices}"
                    )
                return self.gpu_indices_shared.pop()

    async def put_gpu_indices(self, value: str):
        if self.gpu_indices is not None:
            async with self.lock:
                if value not in self.gpu_indices:
                    raise RuntimeError(
                        f"GPU index {value} is not among " f"GPU indices set {self.gpu_indices}"
                    )
                self.gpu_indices_shared.add(value)


class RunStatus(Enum):
    OK = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    process: Optional[asyncio.subprocess.Process] = None


def parse_toml(filepath: str):
    fp = Path(filepath).expanduser().resolve()
    if not fp.exists():
        raise RuntimeError(f"File {fp} does not exist")
    toml_dict: Mapping = toml.load(fp)
    return Config(toml_dict)


@dataclass
class StatusPrinter:
    cmd: str
    flush: bool = True

    def print_skip(self):
        skip_text = colored("[SKIP]", "grey", attrs=["bold"])
        status = colored("[COMPLETED]", "grey")
        self._print_message(f"{skip_text} {status}")

    def print_run(self, pid: int):
        status = colored("[RUN]", "blue", attrs=["bold"])
        self._print_message(f"{self._pid_header(pid)} {status}")

    def print_success(self, pid: int):
        status = colored("[SUCCESS]", "green", attrs=["bold"])
        self._print_message(f"{self._pid_header(pid)} {status}")

    def print_failure(self, pid: int):
        status = colored("[FAILED]", "red", attrs=["bold"])
        self._print_message(f"{self._pid_header(pid)} {status}")

    def _print_message(self, header: str):
        cmd = colored(self.cmd, "grey", attrs=["dark"])
        msg = f"{header} {cmd}"
        print(msg, flush=self.flush)

    def _pid_header(self, pid: int) -> str:
        pid_str = colored(str(pid), "grey", attrs=["bold"])
        return colored(f"[PID:{pid_str}]", "grey")


async def run_command(ctx: RunContext, exp: Unit):
    sp = StatusPrinter(exp.cmd)
    uid = exp.uid
    if ctx.is_task_completed(uid):
        sp.print_skip()
        return RunResult(RunStatus.SKIPPED, None)

    gpu_indices = await ctx.pop_gpu_indices()
    env = deepcopy(os.environ)
    if gpu_indices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_indices)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    stdout_pipe = asyncio.subprocess.PIPE
    read_pipe, write_pipe = os.pipe()
    read_err_pipe, write_err_pipe = os.pipe()

    program, *args = ctx.program_and_args(exp.cmd)
    process = await asyncio.create_subprocess_exec(
        program,
        *args,
        stdout=write_pipe,
        stderr=write_err_pipe,
        env=env,
    )

    os.close(write_pipe)
    os.close(write_err_pipe)

    tee_out = await asyncio.create_subprocess_exec(
        "tee", "-a", ctx.output_log_file(uid), stdin=read_pipe, stdout=stdout_pipe
    )
    tee_err = await asyncio.create_subprocess_exec(
        "tee", "-a", ctx.error_log_file(uid), stdin=read_err_pipe, stdout=stdout_pipe
    )

    os.close(read_pipe)
    os.close(read_err_pipe)

    pid = process.pid
    sp.print_run(pid)

    stdout, stderr = await process.communicate()
    await tee_out.communicate()
    await tee_err.communicate()

    if process.returncode != 0:
        sp.print_failure(pid)
        status = RunStatus.FAILED
    else:
        sp.print_success(pid)
        completed_file = ctx.completed_file(uid)
        # Blocking operations
        completed_file.parent.mkdir(parents=True, exist_ok=True)
        completed_file.touch()
        status = RunStatus.OK

    await ctx.put_gpu_indices(gpu_indices)
    return RunResult(status, process)


async def run(config: Config) -> Sequence[RunResult]:
    max_concurrency: int = config.flags.num_proc
    semaphore = asyncio.BoundedSemaphore(max_concurrency)
    context = RunContext(config)

    async def with_semaphore(ctx, exp):
        async with semaphore:
            res = await run_command(ctx, exp)
            return res

    tasks = []
    for experiment_unit in config.gen_experiments():
        task = asyncio.create_task(with_semaphore(context, experiment_unit))
        tasks.append(task)

    return await asyncio.gather(*tasks, return_exceptions=True)


def exec_toml_config(toml_filepath: str):
    config = parse_toml(toml_filepath)
    loop = asyncio.get_event_loop()
    try:
        results = loop.run_until_complete(run(config))
        _print_final_msg(results)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def _print_final_msg(results):
    ok = _getstatus(results, RunStatus.OK)
    failed = _getstatus(results, RunStatus.FAILED)
    skipped = _getstatus(results, RunStatus.SKIPPED)
    count_fn = lambda x: f"[{len(x)}/{len(results)}]"
    header = colored(f"Job is done!", "grey", attrs=["bold"])
    succeded = colored(f"Successfully completed {count_fn(ok)}", "green")
    failed = colored(f"Failed {count_fn(failed)}", "red")
    skipped = colored(f"Skipped {count_fn(skipped)}", "grey", attrs=["dark"])
    print(f"\r\n{header}\r\n{succeded}\r\n{failed}\r\n{skipped}\r\n")


def _getstatus(results: Sequence[RunResult], status: RunStatus) -> Sequence[RunResult]:
    filtered = filter(lambda x: x.status == status, results)
    return list(filtered)


@click.command()
@click.argument("config-filepath", type=click.Path(exists=True))
def main(config_filepath: str):
    exec_toml_config(config_filepath)


if __name__ == "__main__":
    main()