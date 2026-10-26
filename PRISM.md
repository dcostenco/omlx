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

## Owned image requests

The authenticated endpoint
`/prism-control/v1/models/{model_id}/image-input` reports whether the exact model
and current server configuration support the `owned-png-v1` input contract.
A model's name or vision tag alone does not establish support. The server
captures the applicable settings and refuses changed settings, unsupported
configuration or concurrent model work. Request fields cannot enable support.

Supported input consists of inline PNG images in user messages. Remote URLs,
filesystem paths, tool-output images, other formats and malformed images are
refused. The limits are four images, 128 KiB per image, 192 KiB combined,
2048 pixels per side, 4,194,304 pixels per image and 8,388,608 combined pixels.
The 256 KiB serialized HTTP body limit also applies, including base64 expansion
and the rest of the request. Validation does not resize or convert images.

An image request reserves its model exclusively until backend stop is proven.
Ordinary and other owned acquisitions return busy while that reservation is
held. Cancellation or a caller timeout does not by itself prove that backend
work stopped. Unresolved work retains its reservation and may require recovery;
it must not be treated as idle or safely evictable. Text-only requests retain
their existing behavior.

The capability response describes accepted input, not visual accuracy or
universal memory fit. Clients must retain their resource checks and handle
refusal. Image validation and decoding can currently run on the API event loop.
Keep requests within the published limits. Server capability discovery does not
activate or update a client installation.

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

## Side-by-side managed servers

Set `OMLX_CLUSTER_SHIM_DIRECTORY` to an absolute private directory when starting
a secondary server. Its `omlx-cluster-python` launcher is written there instead
of replacing `~/.omlx/bin/omlx-cluster-python`. The default path is unchanged
when the variable is unset. SSH discovery of the secondary server requires its
explicit launcher path; the default SSH probe continues to select the primary
installation. Relative directories are refused without falling back to the
shared path.

To restrict discovery to the secondary server's configured model directories,
set `"huggingface": {"hf_cache_enabled": false}` in its settings. A private
`HF_HUB_CACHE` also keeps its runtime Hub operations separate from the primary
installation's cache.

## Known issue (upstream)

After a request is requeued for memory and its resumed prefill fails and is
retried, the final output can drop the part folded into the prompt. The path
was observed on the earlier release candidate. This upgrade has not been
qualified against that condition; do not assume stable resolves it.
