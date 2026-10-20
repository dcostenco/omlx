# oMLX for Prism

This fork of [oMLX](https://github.com/jundot/omlx) adds the control routes that
[Prism](https://github.com/dcostenco/prism-coder)'s opt-in oMLX backend needs
(`prism_infer` with `local_backend: "omlx"`, Prism 20.22.0 and later). Stock
oMLX does not have them, so Prism refuses every explicit oMLX call against it.

It is upstream `v0.7.1` stable with the existing Prism control integration
carried forward. The same Apache-2.0 license applies.

## What it adds

- **Owned jobs.** `/prism-control/v1/jobs/{job_id}/operations/{operation_id}`
  routes register, execute, read and cancel a request, behind the server's API
  key. Prism runs each generation as an owned job through them, so a timeout or
  cancel is reported only once the server confirms the request stopped.
- **Owned admission.** An owned request is refused, not served some other way,
  when its model cannot run as configured: another variant is loaded, the model
  is still loading, or an unload is pending.
- **No prompt cache on request.** A registered request with
  `prompt_cache_retention: "none"` skips the completion-time cache store; Prism
  sends it for classifier reads, which are read once.
- **Flash-Next (`qwen4_exp`).** Owned admission for this model family.

## Install

From source, as upstream describes, from this fork's stable integration branch after it is published:

```bash
git clone --branch prism-v0.7.1-stable-20261020 https://github.com/dcostenco/omlx.git
cd omlx
make install                # or: make install-no-kernels without full Xcode
```

Requirements are upstream's: macOS 15.0+, Python 3.11–3.13, Apple Silicon.

Then, for Prism:

1. **Set an API key** in oMLX and keep the server bound to loopback
   (`127.0.0.1`, the default). The control routes exist only when both hold;
   otherwise Prism refuses every call. Prism reads the key from `OMLX_API_KEY`
   or `~/.omlx/settings.json`.
2. **Turn on chunked prefill**: `"scheduler": {"chunked_prefill": true}` in
   `~/.omlx/settings.json`. Without it, a cancel during a long prompt read
   cannot be confirmed, and Prism refuses later calls (`omlx_stop_unproved`,
   then `omlx_provider_busy`).
3. **Install Prism's classifier**: the public
   `dcostenco/prism-coder-9b-mlx-8bit`, in `~/.omlx/models/prism-coder-9b-mlx-8bit`
   (Prism refers to it by that name). It holds about 9.2 GB while loaded.
4. **Give each drafting model a thinking budget** in its model settings (for
   example 256 tokens). Prism leaves thinking to oMLX's settings, so a model
   with no budget can answer with reasoning only, and Prism refuses that
   (`think_only`).
5. **Keep the normal soft memory watermark.** For example, a custom ceiling
   of 27 GB with `soft_threshold: 0.90` lets oMLX evict the idle 9B classifier
   before loading a 27B 4-bit drafting model. Owned completion now retires
   the consumed request's collector after all stop proofs, so a classifier
   that has just answered is immediately eligible for idle eviction. Raising
   the soft threshold to keep both models resident can cause memory churn
   during loading. The soft watermark controls idle eviction; the hard
   ceiling and Prism's resource probes still apply.

## Known issue (upstream)

After a request is requeued for memory and its resumed prefill fails and is
retried, the final output can drop the part folded into the prompt. The path
was observed on the earlier release candidate. This upgrade has not been
qualified against that condition; do not assume stable resolves it.
