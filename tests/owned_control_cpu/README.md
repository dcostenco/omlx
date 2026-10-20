# Owned-control CPU regressions

Run from any checkout, without a model, local daemon, or private evaluation directory:

```sh
python tests/owned_control_cpu/run.py
```

The normal pytest suite invokes this runner through `tests/test_owned_control_cpu.py`,
so the existing pull-request CI runs these regressions too. Child-process isolation
keeps the suite's deliberate MLX/server import refusal and CPU module fixtures out
of pytest's shared process. Unexpected skips and an empty suite fail the runner.

The suite checks registration, ownership, cancellation, durable terminal records,
admission, cleanup, stream barriers, and the shared Responses preparation. Executor
and scheduler boundaries use the checkout's actual method bodies with CPU fixtures;
this does not establish real-GPU stop behavior or model quality.

The three `omlx_*` fixtures preserve earlier Apache-2.0 source bodies for differential
checks. Despite the historical filename, `omlx_create_response_0032173.py.txt` contains
the v0.7.1 stable `bd1e9b112474862ebed550a9c83b4316038a1d74` Responses baseline.
