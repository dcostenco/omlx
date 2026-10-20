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

The Apache-2.0 fixture `responses_v0_7_1_bd1e9b1.py.txt` preserves the Responses
route's signature and body from upstream v0.7.1 stable commit
`bd1e9b112474862ebed550a9c83b4316038a1d74`, with the HTTP decorator omitted.
The consuming test pins its SHA256 before comparing preparation bodies. The two
`omlx_*` fixtures preserve earlier engine-method bodies for differential checks.
