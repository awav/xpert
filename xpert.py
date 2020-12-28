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
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Generator, List, Mapping, MutableSet, Optional

import toml
from termcolor import colored


def is_sequence(v: Any) -> bool:
    return not isinstance(v, str) and isinstance(v, Sequence)


def is_xprod(v: Any) -> bool:
    if is_sequence(v):
        return True
    return False


def is_option_settings(opt: Any) -> bool:
    if not isinstance(opt, dict):
        return False
    keys = set(opt.keys())
    return keys.issubset(["value", "type", "xprod"])


def preprocess(opts: Dict) -> Dict:
    opts_processed = deepcopy(opts)
    for key, val in opts.items():
        if is_option_settings(val):
            value = val["value"]
            value_type = val.get("type", None)
            value_xprod = val.get("xprod", False)

            if value_type.lower() == "path":
                value = [Path(v).resolve() for v in glob.glob(value)]

            if not value_xprod:
                pass

            opts_processed[key] = value
    return opts_processed


def gen_xprod(d: Dict) -> Generator:
    keys = d.keys()
    vals = d.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def opts_configurations(raw_cfg: Mapping) -> List[Dict]:
    opts = preprocess(raw_cfg["opts"])
    included = {k: v for k, v in opts.items() if is_xprod(v)}
    excluded = {k: v for k, v in opts.items() if not is_xprod(v)}
    opts_all = [{**excluded, **d} for d in gen_xprod(included)]
    return opts_all


def _norm_template(template: str) -> str:
    pattern = r"{opts.([\w-]*)}"
    replacement = r"{opts_\1}"
    return re.sub(pattern, replacement, template)


def _norm_opts(opts: Dict) -> Dict:
    return {f"opts_{k}": v for k, v in opts.items()}


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


@dataclass(frozen=True)
class RunConfig:
    cmd_tmpl: str
    uid_tmpl: str
    options: List[Dict]
    proc_num: int = 4
    restart: bool = False
    gpu_indices: Optional[FrozenSet[str]] = None


@dataclass
class RunContext:
    lock: asyncio.Lock
    config: InitVar[RunConfig]

    _restart: bool = field(init=False)
    _uid_tmpl: str = field(init=False)
    _cmd_tmpl: str = field(init=False)
    _gpu_indices: Optional[FrozenSet[str]] = field(init=False, default=None)
    _gpu_indices_shared: Optional[MutableSet[str]] = field(init=False, default=None)

    def __post_init__(self, config: RunConfig):
        self._restart = config.restart
        self._uid_tmpl = _norm_template(config.uid_tmpl)
        self._cmd_tmpl = _norm_template(config.cmd_tmpl)
        if config.gpu_indices is not None:
            indices = config.gpu_indices
            self._gpu_indices = frozenset(deepcopy(indices))
            self._gpu_indices_shared = set(deepcopy(indices))

    def cmd(self, opts: Dict) -> str:
        opts_norm = _norm_opts(opts)
        uid = self.uid(opts)
        cmd = self._cmd_tmpl.format(uid=uid, **opts_norm)
        return cmd

    def uid(self, opts: Dict) -> str:
        opts_norm = _norm_opts(opts)
        opts_values = sorted(list(map(str, opts_norm.values())))
        to_hash = ".".join(opts_values)
        md5 = hashlib.md5(to_hash.encode("utf-8")).hexdigest()
        return self._uid_tmpl.format(hash_opts=md5, **opts_norm)

    def program_and_args(self, opts: Dict) -> List[str]:
        return _quoted_split(self.cmd(opts))

    def output_log_file(self, opts: Dict) -> Path:
        timestamp = int(time.time())
        return self._filepath(f"stdout.{timestamp}.log", opts)

    def error_log_file(self, opts: Dict) -> Path:
        timestamp = int(time.time())
        return self._filepath(f"stderr.{timestamp}.log", opts)

    def completed_file(self, opts: Dict) -> Path:
        return self._filepath("completed.status", opts)

    def failed_file(self, opts: Dict) -> Path:
        return self._filepath("failed.status", opts)

    def _filepath(self, status: str, opts: Dict) -> Path:
        base = Path(self.uid(opts))
        if not base.exists():
            base.mkdir(parents=True)
        return Path(base, status).resolve()

    def is_task_completed(self, opts: Dict) -> bool:
        if self._restart:
            return False
        return self.completed_file(opts).exists()

    async def pop_gpu_indices(self):
        if self._gpu_indices is not None:
            async with self.lock:
                if len(self._gpu_indices_shared) == 0:
                    raise RuntimeError(
                        f"GPU indices set is empty." f"Original set: {self._gpu_indices}"
                    )
                return self._gpu_indices_shared.pop()

    async def put_gpu_indices(self, value: Optional[str]):
        if self._gpu_indices is not None:
            async with self.lock:
                if value not in self._gpu_indices:
                    raise RuntimeError(
                        f"GPU index {value} is not among " f"GPU indices set {self._gpu_indices}"
                    )
                self._gpu_indices_shared.add(value)


@dataclass(frozen=True)
class RunResult:
    process: Optional[asyncio.subprocess.Process] = None


def _cli_name() -> str:
    return "xpert"


def _default_uid() -> str:
    return f"{_cli_name()}.logs/{{hash_opts}}"


def parse_toml(filepath: str):
    fp = Path(filepath).expanduser().resolve()
    if not fp.exists():
        raise RuntimeError(f"File {fp} does not exist")
    raw_cfg: Mapping = toml.load(fp)
    opts = opts_configurations(raw_cfg)
    cmd_tmpl = raw_cfg["cmd"]
    uid_tmpl = str(raw_cfg.get("uid", _default_uid()))
    proc_cfg = raw_cfg.get("proc", dict())
    gpu_cfg = proc_cfg.get("gpu", dict())
    proc_num = proc_cfg.get("num", RunConfig.proc_num)
    gpu_indices = gpu_cfg.get("indices", RunConfig.gpu_indices)
    return RunConfig(cmd_tmpl, uid_tmpl, opts, proc_num=proc_num, gpu_indices=gpu_indices)


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


async def run_command(ctx: RunContext, opts: Dict):
    sp = StatusPrinter(ctx.cmd(opts))
    if ctx.is_task_completed(opts):
        sp.print_skip()
        return RunResult()

    gpu_indices = await ctx.pop_gpu_indices()
    env = deepcopy(os.environ)
    if gpu_indices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_indices)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    stdout_pipe = asyncio.subprocess.PIPE
    read_pipe, write_pipe = os.pipe()
    read_err_pipe, write_err_pipe = os.pipe()

    program, *args = ctx.program_and_args(opts)
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
        "tee", "-a", ctx.output_log_file(opts), stdin=read_pipe, stdout=stdout_pipe
    )
    tee_err = await asyncio.create_subprocess_exec(
        "tee", "-a", ctx.error_log_file(opts), stdin=read_err_pipe, stdout=stdout_pipe
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
    else:
        sp.print_success(pid)
        completed_file = ctx.completed_file(opts)
        # Blocking operations
        completed_file.parent.mkdir(parents=True, exist_ok=True)
        completed_file.touch()

    await ctx.put_gpu_indices(gpu_indices)
    return RunResult(process)


async def run(config: RunConfig):
    max_concurrency = config.proc_num
    semaphore = asyncio.BoundedSemaphore(max_concurrency)
    ctx = RunContext(asyncio.Lock(), config)

    async def with_semaphore(ctx, opts):
        async with semaphore:
            res = await run_command(ctx, opts)
            return res

    tasks = []
    for opts in config.options:
        task = asyncio.create_task(with_semaphore(ctx, opts))
        tasks.append(task)

    return await asyncio.gather(*tasks, return_exceptions=True)


def exec_toml_config(toml_filepath: str):
    config = parse_toml(toml_filepath)
    loop = asyncio.get_event_loop()
    try:
        result = loop.run_until_complete(run(config))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


@click.command()
@click.argument("config-filepath", type=click.Path(exists=True))
def main(config_filepath: str):
    exec_toml_config(config_filepath)


if __name__ == "__main__":
    main()