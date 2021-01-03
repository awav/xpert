# Xpert

Simple experiment runner tool:

* Specifying jobs via a configuration TOML file
* Running jobs in parallel
* Redirecting `stdout` and `stderr` into a separate files
* Tracking job completion
* Imposing restrictions: specifying GPU indexes or a number of parallel processes

In the following TOML configuration example we run `python -c ...` command in :


```toml
cmd = 'import tensorflow as tf; import os; print(tf.random.normal([4, 4]) + tf.eye(4));'
uid = "xpert.logs/{index}"  # "{KEY}" is a formatting syntax. KEY will be replaced with an option from an experiment setup

[[exp]]  # First group of experiments
cmd = "python -c 'print(\"job-{index}\", \"group-{name}\"); {cmd}'"
name = "group.1"  # Literal
index = [0, 1]  # List specifies cross-product points for experiment generation

[[exp]]  # Second group of experiments
cmd = "python -c 'print(\"job-{index}\", \"group-{name}\"); {cmd}'"
name = '"group.2"'
index = [10, 20]

[flags]
restart = false
num_proc = 2
gpu_indices = ["1", "2,4"]

```

When you run the script you will get:

```
→  python xpert.py config-example.toml
[PID:16800] [RUN] python -c 'import tensorflow as tf; import os; print(tf.random.normal([4, 4]) + tf.eye(4)); print(os.environ.get("CUDA_VISIBLE_DEVICES", "NOTFOUND")); print("0", "group.1")'
[PID:16801] [RUN] python -c 'import tensorflow as tf; import os; print(tf.random.normal([4, 4]) + tf.eye(4)); print(os.environ.get("CUDA_VISIBLE_DEVICES", "NOTFOUND")); print("1", "group.1")'
[PID:16801] [SUCCESS] python -c 'import tensorflow as tf; import os; print(tf.random.normal([4, 4]) + tf.eye(4)); print(os.environ.get("CUDA_VISIBLE_DEVICES", "NOTFOUND")); print("1", "group.1")'

...

Job is done!
Successfully completed [6/6]
Failed [0/6]
Skipped [0/6]
```

And the output directory will contain:

```
→ ls xpert.logs/1/
completed.flag
stdout.1609672971.log
stderr.1609672971.log
```
