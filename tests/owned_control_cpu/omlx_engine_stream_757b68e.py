# SPDX-License-Identifier: Apache-2.0
# Verbatim stream_chat and stream_generate of BatchedEngine and
# VLMBatchedEngine from oMLX 757b68e, before slice 27.4a split them at
# add_request. The differential baseline for
# test_provider_owned_admission_plumbing.py; never edited by hand.
from __future__ import annotations


class BaselineBatchedEngine:
    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text or token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            min_p: Min-p sampling (0.0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            presence_penalty: Presence penalty (0.0 = disabled)
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        from ..request import SamplingParams

        self._prepare_k2_tool_grammar(kwargs.get("tools"), kwargs)
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            repetition_penalty=repetition_penalty,
            repetition_context_size=kwargs.get("repetition_context_size"),
            presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [],
            thinking_budget=kwargs.get("thinking_budget", None),
            compiled_grammar=kwargs.get("compiled_grammar", None),
            seed=kwargs.get("seed", None),
            stop_skips_reasoning=kwargs.get("stop_skips_reasoning", False),
        )

        # SpecPrefill: pass per-request overrides to engine
        specprefill_kwargs = self._pop_specprefill_kwargs(kwargs)
        tools = kwargs.pop("tools", None)

        engine = self._engine
        request_id = await engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=kwargs.pop("_request_id", None),
            tools=tools,
            skip_cache_store=bool(kwargs.get("skip_cache_store", False)),
            preserve_reasoning=bool(kwargs.get("preserve_reasoning", False)),
            benchmark_trace=bool(kwargs.get("benchmark_trace", False)),
            benchmark_ane_sequence_length=int(
                kwargs.get("benchmark_ane_sequence_length", 0) or 0
            ),
            **specprefill_kwargs,
        )

        finished_normally = False
        try:
            async for output in engine.stream_outputs(request_id):
                text = clean_special_tokens(output.output_text)

                # Set finished_normally BEFORE yield, because the consumer
                # may stop iterating after receiving the final output,
                # which triggers GeneratorExit at the yield point -
                # code after yield would never execute.
                if output.finished:
                    finished_normally = True

                yield GenerationOutput(
                    text=text,
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    tool_calls=output.tool_calls,
                    cached_tokens=output.cached_tokens,
                    generated_at=getattr(output, "generated_at", None),
                    generated_until=getattr(output, "generated_until", None),
                    first_token_at=getattr(output, "first_token_at", None),
                    benchmark_prefill_chunks=(
                        list(chunks)
                        if (chunks := getattr(output, "benchmark_prefill_chunks", []))
                        else []
                    ),
                    benchmark_requested_steps=(
                        list(steps)
                        if (steps := getattr(output, "benchmark_requested_steps", []))
                        else []
                    ),
                    benchmark_boundary_enabled=bool(
                        getattr(output, "benchmark_boundary_enabled", False)
                    ),
                    benchmark_cache_block_size=int(
                        getattr(output, "benchmark_cache_block_size", 0) or 0
                    ),
                )
        except GeneratorExit:
            # Client disconnected
            logger.info(
                f"[stream_generate] GeneratorExit caught for request {request_id}"
            )
        finally:
            # Abort the request if client disconnected before completion
            if not finished_normally:
                logger.info(
                    f"[stream_generate] Aborting request {request_id} (finished_normally={finished_normally})"
                )
                await engine.abort_request(request_id)
            else:
                logger.debug(
                    f"[stream_generate] Request {request_id} finished normally"
                )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            min_p: Min-p sampling (0.0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            presence_penalty: Presence penalty (0.0 = disabled)
            tools: Optional tool definitions
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        # Preprocess messages for Harmony (gpt-oss) models
        messages = self._preprocess_messages(messages)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        partial = kwargs.pop("is_partial", None)
        prepared = kwargs.pop("_prepared_prompt", None)
        prompt = (
            prepared[0]
            if prepared is not None
            else self._apply_chat_template(
                messages,
                template_tools,
                chat_template_kwargs=ct_kwargs,
                is_partial=partial,
            )
        )

        # SpecPrefill: protect the system-prompt region from token dropping.
        self._inject_specprefill_system_end(
            messages,
            prompt,
            template_tools,
            ct_kwargs,
            kwargs,
            full_prompt_token_count=len(prepared[1]) if prepared is not None else None,
        )
        generation_prompt, persists = self._generation_prompt_text(ct_kwargs, partial)
        if generation_prompt:
            kwargs["generation_prompt_text"] = generation_prompt
            kwargs["generation_prompt_persists"] = persists

        async for output in self.stream_generate(
            prompt=prepared[1] if prepared is not None else prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            tools=tools,
            **kwargs,
        ):
            yield output



class BaselineVLMBatchedEngine:
    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        vlm_inputs_embeds: Any = None,
        vlm_extra_kwargs: dict[str, Any] | None = None,
        vlm_image_hash: str | None = None,
        vlm_cache_key_start: int = 0,
        vlm_cache_key_ranges: Optional[List[Tuple[int, str]]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generation token by token."""
        if not self._loaded:
            await self.start()

        if self.is_diffusion_model:
            if (
                vlm_inputs_embeds is not None
                or vlm_extra_kwargs is not None
                or vlm_image_hash is not None
                or vlm_cache_key_ranges is not None
                or vlm_cache_key_start
            ):
                raise InvalidRequestError(
                    "Precomputed VLM embeddings and cache metadata are not "
                    "supported with diffusion models."
                )
            self._validate_diffusion_request(
                stop=stop,
                kwargs=kwargs,
            )
            loop = asyncio.get_running_loop()
            from ..engine_core import get_mlx_executor

            diffusion_inputs = await loop.run_in_executor(
                get_mlx_executor(),
                self._prepare_diffusion_inputs_from_prompt,
                prompt,
            )
            async for output in self._stream_diffusion_inputs(
                diffusion_inputs,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=kwargs.get("seed"),
            ):
                yield output
            return

        # OCR models: add extra stop token IDs to prevent degeneration.
        # Sampling params (temperature, repetition_penalty, max_tokens) are
        # resolved by get_sampling_params() with OCR defaults as a fallback
        # layer, so admin/API overrides are respected.
        extra_stop_ids: list[int] = []
        if self.is_ocr_model:
            extra_stop_ids = self._resolve_ocr_stop_token_ids()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            repetition_penalty=repetition_penalty,
            repetition_context_size=kwargs.get("repetition_context_size"),
            presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [],
            stop_token_ids=extra_stop_ids or None,
            thinking_budget=kwargs.get("thinking_budget", None),
            compiled_grammar=kwargs.get("compiled_grammar", None),
            seed=kwargs.get("seed", None),
            stop_skips_reasoning=kwargs.get("stop_skips_reasoning", False),
        )

        # SpecPrefill: pass per-request overrides
        specprefill_kwargs = self._pop_specprefill_kwargs(kwargs)
        tools = kwargs.pop("tools", None)

        engine = self._engine
        request_id = await engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=kwargs.pop("_request_id", None),
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
            vlm_cache_key_start=vlm_cache_key_start,
            vlm_cache_key_ranges=vlm_cache_key_ranges,
            skip_cache_store=bool(kwargs.get("skip_cache_store", False)),
            preserve_reasoning=bool(kwargs.get("preserve_reasoning", False)),
            benchmark_trace=bool(kwargs.get("benchmark_trace", False)),
            benchmark_ane_sequence_length=int(
                kwargs.get("benchmark_ane_sequence_length", 0) or 0
            ),
            tools=tools,
            **specprefill_kwargs,
        )

        finished_normally = False
        try:
            async for output in engine.stream_outputs(request_id):
                text = clean_special_tokens(output.output_text)

                if output.finished:
                    finished_normally = True

                yield GenerationOutput(
                    text=text,
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    tool_calls=output.tool_calls,
                    cached_tokens=output.cached_tokens,
                    generated_at=getattr(output, "generated_at", None),
                    generated_until=getattr(output, "generated_until", None),
                    first_token_at=getattr(output, "first_token_at", None),
                    benchmark_prefill_chunks=(
                        list(chunks)
                        if (chunks := getattr(output, "benchmark_prefill_chunks", []))
                        else []
                    ),
                    benchmark_requested_steps=(
                        list(steps)
                        if (steps := getattr(output, "benchmark_requested_steps", []))
                        else []
                    ),
                    benchmark_boundary_enabled=bool(
                        getattr(output, "benchmark_boundary_enabled", False)
                    ),
                    benchmark_cache_block_size=int(
                        getattr(output, "benchmark_cache_block_size", 0) or 0
                    ),
                )
        except GeneratorExit:
            logger.info(f"[vlm_stream_generate] GeneratorExit for request {request_id}")
        finally:
            if not finished_normally:
                logger.info(f"[vlm_stream_generate] Aborting request {request_id}")
                await engine.abort_request(request_id)

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream chat completion with vision support."""
        if not self._loaded:
            await self.start()

        if self.is_diffusion_model:
            self._validate_diffusion_request(
                tools=tools,
                stop=kwargs.get("stop"),
                kwargs=kwargs,
            )
            loop = asyncio.get_running_loop()
            from ..engine_core import get_mlx_executor

            diffusion_inputs = await loop.run_in_executor(
                get_mlx_executor(),
                self._process_diffusion_chat_messages,
                messages,
                tools,
                dict(kwargs),
            )
            async for output in self._stream_diffusion_inputs(
                diffusion_inputs,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=kwargs.get("seed"),
            ):
                yield output
            return

        # Run vision encoding on the MLX executor thread to avoid blocking
        # the event loop.  Blocking here (synchronous mx.eval) prevents
        # uvicorn from managing HTTP keep-alive connections, causing
        # TransferEncodingError on the next request (issue #80).
        loop = asyncio.get_running_loop()
        # _process_chat_messages pops these; the tail marker needs them too.
        ct_kwargs = kwargs.get("chat_template_kwargs")
        partial = kwargs.get("is_partial")
        (
            prompt,
            vlm_embeds,
            vlm_kwargs,
            image_hash,
            image_cache_key_start,
            image_cache_key_ranges,
        ) = await loop.run_in_executor(
            self._engine._mlx_executor,
            self._process_chat_messages,
            messages,
            tools,
            kwargs,
        )

        # SpecPrefill: protect the system-prompt region from token dropping.
        self._inject_specprefill_system_end(messages, prompt, kwargs)
        generation_prompt, persists = self._generation_prompt_text(ct_kwargs, partial)
        if generation_prompt:
            kwargs["generation_prompt_text"] = generation_prompt
            kwargs["generation_prompt_persists"] = persists

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            vlm_inputs_embeds=vlm_embeds,
            vlm_extra_kwargs=vlm_kwargs,
            vlm_image_hash=image_hash,
            vlm_cache_key_start=image_cache_key_start,
            vlm_cache_key_ranges=image_cache_key_ranges,
            tools=tools,
            **kwargs,
        ):
            yield output
