# oMLX for Prism

This fork of [oMLX](https://github.com/jundot/omlx) adds the control routes that
[Prism](https://github.com/dcostenco/prism-coder)'s opt-in oMLX backend needs
(`prism_infer` with `local_backend: "omlx"`, Prism 20.22.0 and later). Stock
oMLX does not have them, so Prism refuses every explicit oMLX call against it.

It is upstream `v0.7.1rc1`, a pre-release, plus one commit. It will move to
0.7.1 stable when that is released. Everything else is upstream oMLX, under
the same Apache-2.0 license.

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

From source, as upstream describes, from this fork's `prism-v0.7.1rc1` branch:

```bash
git clone --branch prism-v0.7.1rc1 https://github.com/dcostenco/omlx.git
cd omlx
make install                # or: make install-no-kernels without full Xcode
```

Requirements are upstream's: macOS 15.0+, Python 3.11–3.13, Apple Silicon.

Then, for Prism:

1. **Set an API key** in oMLX. Prism reads it from `OMLX_API_KEY` or
   `~/.omlx/settings.json`.
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
5. **Leave room for the classifier beside your drafting model.** oMLX evicts an
   idle model to fit another, but not one that has just answered. Measured on
   a 48 GB Mac with a 27B 4-bit drafting model: a custom memory limit of 27 GB
   with a 0.97 soft threshold kept both loaded for normal prompts; at 24 and
   26 GB most drafts were refused.

## Known issue (upstream)

After a request is requeued for memory and its resumed prefill fails and is
retried, the final output can drop the part folded into the prompt. The path
is rare and affects any request, owned or not; it is upstream code, unchanged
here. Check whether 0.7.1 stable fixes it before relying on this build.
