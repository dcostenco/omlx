# SPDX-License-Identifier: Apache-2.0
"""Slice 28: the host-read boundaries owned Flash-Next work relies on.

Gather-ahead off at the one SSD-backed PLE submission line, and preserved
when on; every host-read submission counted, including one that enqueues its
read and then raises (``ThreadPoolExecutor.submit`` starts its worker after
enqueueing); the MoE partial-group drain.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from omlx import host_submit  # noqa: E402
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat  # noqa: E402
from omlx.patches import moe_expert_offload as offload  # noqa: E402
from test_mlx_vlm_qwen4_exp_compat import _disk_ple, _tiny_config, disk_ple_reader  # noqa: E402,F401


class EnqueueThenRaise:
    """``ThreadPoolExecutor.submit`` in order: enqueue the work item, then start
    a worker. At submission ``fail_at`` the start fails: the read is queued and
    runs later, but the caller gets the exception, not the future. Earlier
    submissions run at once on the one worker, before the hidden read."""

    def __init__(self, fail_at):
        self.inner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="enqueue-then-raise")
        self.fail_at = fail_at
        self.submissions = 0
        self.release = threading.Event()
        self.hidden_done = threading.Event()

    def submit(self, fn, /, *args, **kwargs):
        self.submissions += 1
        if self.submissions == self.fail_at:
            def hidden():
                self.release.wait(10)
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.hidden_done.set()

            self.inner.submit(hidden)
            raise RuntimeError("can't start new thread")
        return self.inner.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True, cancel_futures=False):
        self.release.set()
        self.inner.shutdown(wait=wait, cancel_futures=cancel_futures)


def _language():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    return language


# --- PLE gather-ahead --------------------------------------------------------


def test_configure_ple_runtime_captures_gather_ahead_for_the_next_tables(tmp_path, monkeypatch):
    language = _language()
    for name in ("_PLE_RUNTIME_MODEL_PATH", "_PLE_RUNTIME_MODE", "_PLE_RUNTIME_GATHER_AHEAD"):
        monkeypatch.setattr(language, name, getattr(language, name))
    language.configure_ple_runtime(tmp_path, mode="mmap", gather_ahead=False)
    assert language._PLE_RUNTIME_GATHER_AHEAD is False
    embedding, _ = _disk_ple(tmp_path, shards=2, rows=8, dims=32, bits=4)
    assert embedding.gather_ahead is False
    embedding.close()
    language.configure_ple_runtime(tmp_path, mode="mmap")  # default stays on
    assert language._PLE_RUNTIME_GATHER_AHEAD is True


def test_the_compat_configure_passes_gather_ahead_through(tmp_path, monkeypatch):
    language = _language()
    for name in ("_PLE_RUNTIME_MODEL_PATH", "_PLE_RUNTIME_MODE", "_PLE_RUNTIME_GATHER_AHEAD", "_MTP_RUNTIME"):
        monkeypatch.setattr(language, name, getattr(language, name))
    compat.configure_qwen4_exp_runtime(tmp_path, mode="mmap", ple_gather_ahead=False)
    assert language._PLE_RUNTIME_GATHER_AHEAD is False
    compat.configure_qwen4_exp_runtime(tmp_path, mode="mmap")
    assert language._PLE_RUNTIME_GATHER_AHEAD is True


def test_gather_ahead_off_submits_nothing_for_any_producer_shape(tmp_path, monkeypatch):
    language = _language()
    monkeypatch.setattr(language, "_PLE_RUNTIME_GATHER_AHEAD", False)
    embedding, table = _disk_ple(tmp_path, shards=3, rows=8, dims=64, bits=5)
    assert embedding.gather_ahead is False
    real = embedding._prefetch_executor
    submitted = []
    monkeypatch.setattr(
        embedding,
        "_prefetch_executor",
        SimpleNamespace(submit=lambda *args, **kwargs: submitted.append(args), shutdown=real.shutdown),
    )
    before = host_submit.submission_failures()
    # First-chunk announce, next chunk, a multi-row batch, a single row, empty.
    shapes = ([[3, 17, 5, 22, 3, 9]], [[1, 2]], [[4, 5], [6, 7]], [[7]], [[]])
    for rows in shapes:
        embedding.prefetch(mx.array(rows, dtype=mx.int32).reshape(len(rows), -1))
    assert (submitted, embedding.prefetch_submissions, embedding._pending) == ([], 0, {})
    assert host_submit.submission_failures() == before
    first = mx.array([[3, 17, 5, 22, 3, 9]], dtype=mx.int32)
    values = embedding(first)
    mx.eval(values)
    assert embedding.last_prefetch_hit is False
    assert mx.array_equal(values, mx.take(table, first.reshape(-1), axis=0)[None]).item()
    embedding.close()


def test_gather_ahead_on_still_submits_and_serves_the_same_rows(tmp_path):
    embedding, table = _disk_ple(tmp_path, shards=3, rows=8, dims=64, bits=5)
    assert embedding.gather_ahead is True
    first = mx.array([[3, 17, 5, 22, 3, 9]], dtype=mx.int32)
    cold = embedding(first)
    mx.eval(cold)
    assert embedding.last_prefetch_hit is False
    embedding.prefetch(first)
    assert embedding.prefetch_submissions == 1
    warm = embedding(first)
    mx.eval(warm)
    assert embedding.last_prefetch_hit is True
    assert mx.array_equal(warm, cold).item()
    assert mx.array_equal(warm, mx.take(table, first.reshape(-1), axis=0)[None]).item()
    embedding.close()


def test_ple_gathers_ahead_is_false_for_a_table_with_gather_ahead_off():
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    table = next(
        layer.ple for layer in model.model.layers if getattr(layer, "ple", None) is not None
    ).ple_embedding.ngram_embedding
    table.prefetch = lambda indices: None
    assert model.ple_gathers_ahead() is True
    table.gather_ahead = False
    assert model.ple_gathers_ahead() is False


def test_prefetch_ple_skips_and_never_logs_active_for_a_table_with_gather_ahead_off(caplog):
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    table = next(
        layer.ple for layer in model.model.layers if getattr(layer, "ple", None) is not None
    ).ple_embedding.ngram_embedding
    calls = []
    table.prefetch = lambda indices: calls.append(indices)
    current, following = mx.array([[3, 4, 5]], dtype=mx.int32), mx.array([[6, 7]], dtype=mx.int32)
    table.gather_ahead = False
    with caplog.at_level("INFO"):
        model.prefetch_ple(following, current)
    assert calls == []
    assert "gather-ahead active" not in caplog.text
    table.gather_ahead = True
    with caplog.at_level("INFO"):
        model.prefetch_ple(following, current)
    assert calls  # on: the table is still asked, and the log is honest
    assert "gather-ahead active" in caplog.text


# --- Counted submissions: a read accepted without a future -------------------


def test_gather_ahead_submission_that_enqueues_then_raises_is_counted(tmp_path, monkeypatch):
    embedding, _ = _disk_ple(tmp_path, shards=3, rows=8, dims=64, bits=5)
    real = embedding._prefetch_executor
    executor = EnqueueThenRaise(fail_at=1)
    monkeypatch.setattr(embedding, "_prefetch_executor", executor)
    before = host_submit.submission_failures()
    with pytest.raises(RuntimeError, match="can't start new thread"):
        embedding.prefetch(mx.array([[3, 17, 5]], dtype=mx.int32))
    assert host_submit.submission_failures() == before + 1
    assert embedding._pending == {}  # no future to join...
    assert not executor.hidden_done.is_set()  # ...and the read has not run yet
    executor.release.set()
    assert executor.hidden_done.wait(10)  # it runs anyway
    embedding.close()
    real.shutdown(wait=True)


def test_page_touch_submission_that_enqueues_then_raises_is_counted(disk_ple_reader, monkeypatch):
    ple, reader, dense = disk_ple_reader
    executor = EnqueueThenRaise(fail_at=2)
    monkeypatch.setattr(ple, "_PLE_IO_POOL", executor)
    before = host_submit.submission_failures()
    indices = [0, 1, 7, 9, 1000, 2000, 3000, 4000, 4095]
    with pytest.raises(RuntimeError, match="can't start new thread"):
        reader.rows("weight", indices)
    assert executor.submissions == 2
    assert host_submit.submission_failures() == before + 1
    assert not executor.hidden_done.is_set()
    executor.release.set()
    assert executor.hidden_done.wait(10)
    executor.shutdown()


class _Plans:
    """The part of ExpertCache that _submit and _read_ahead use."""

    _submit = offload.ExpertCache._submit
    _read_ahead = offload.ExpertCache._read_ahead
    _bind_staging = offload.ExpertCache._bind_staging

    def __init__(self):
        self.slot_of = {}
        self._staging = None
        self.disk = SimpleNamespace(_store=SimpleNamespace(_staging=None))

    def _plans(self, e):
        return [(name, 0, ("plan", e, index)) for index, name in enumerate(offload._PROJS)]


@pytest.fixture
def plain_reads(monkeypatch):
    monkeypatch.setattr(offload.CheckpointExpertStore, "read", staticmethod(lambda plan: plan))


def _in_thread(fn):
    box = {}

    def run():
        try:
            box["result"] = fn()
        except BaseException as error:  # noqa: BLE001
            box["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    return thread, box


def test_moe_submit_that_enqueues_then_raises_is_counted(plain_reads):
    executor = EnqueueThenRaise(fail_at=2)
    before = host_submit.submission_failures()
    with pytest.raises(RuntimeError, match="can't start new thread"):
        _Plans()._submit(executor, 5)
    assert host_submit.submission_failures() == before + 1
    assert not executor.hidden_done.is_set()
    executor.release.set()
    assert executor.hidden_done.wait(10)
    executor.shutdown()


def test_moe_owned_submission_uses_stable_staging_reads(monkeypatch):
    """The stable shared-buffer read must retain owned submission tracking."""
    reads = []
    plans = _Plans()
    plans._staging = SimpleNamespace(read=lambda plan: reads.append(plan) or plan)
    monkeypatch.setattr(
        offload.CheckpointExpertStore,
        "read",
        staticmethod(lambda plan: pytest.fail("staging read was bypassed")),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        group = plans._submit(executor, 5)
        assert [future.result(timeout=2) for _, _, _, future in group] == [
            plan for _, _, plan in plans._plans(5)
        ]
    assert reads == [plan for _, _, plan in plans._plans(5)]


def test_moe_partial_group_is_drained_before_a_failed_submission_raises(plain_reads):
    inner = ThreadPoolExecutor(max_workers=1)
    release, first_done = threading.Event(), threading.Event()
    calls = []

    class FirstHeldThenFails:
        def submit(self, fn, /, *args):
            calls.append(args)
            if len(calls) == 1:
                def held():
                    release.wait(10)
                    try:
                        return fn(*args)
                    finally:
                        first_done.set()

                return inner.submit(held)
            raise RuntimeError("cannot schedule new futures")  # before enqueueing

    before = host_submit.submission_failures()
    thread, box = _in_thread(lambda: _Plans()._submit(FirstHeldThenFails(), 7))
    time.sleep(0.2)
    assert thread.is_alive()  # draining the held first read
    release.set()
    thread.join(10)
    assert isinstance(box.get("error"), RuntimeError)
    assert first_done.is_set()
    assert host_submit.submission_failures() == before + 1
    inner.shutdown()


def test_moe_read_ahead_drains_its_partial_dict_before_raising(plain_reads, monkeypatch):
    inner = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    ran, accepted = [], []

    class HeldThenFails:
        submitted = 0

        def submit(self, fn, /, *args):
            HeldThenFails.submitted += 1
            if HeldThenFails.submitted > len(offload._PROJS):  # expert 2's first read
                raise RuntimeError("cannot schedule new futures")

            def held():
                release.wait(10)
                ran.append(args)
                return fn(*args)

            accepted.append(inner.submit(held))
            return accepted[-1]

    monkeypatch.setattr(offload, "_io_pool", lambda: HeldThenFails())
    thread, box = _in_thread(lambda: _Plans()._read_ahead([1, 2], limit=4))
    time.sleep(0.2)
    assert thread.is_alive()  # waiting on expert 1's running read
    release.set()
    thread.join(10)
    assert isinstance(box.get("error"), RuntimeError)
    # Every read expert 1 was granted is finished or cancelled before the
    # error reaches the caller (_drain cancels queued reads, waits for the rest).
    assert len(accepted) == len(offload._PROJS)
    assert all(future.done() for future in accepted)
    assert len(ran) + sum(future.cancelled() for future in accepted) == len(accepted)
    inner.shutdown(wait=True)
    assert len(ran) + sum(future.cancelled() for future in accepted) == len(accepted)  # none ran late


def test_moe_submit_without_failures_is_unchanged(plain_reads):
    before = host_submit.submission_failures()
    with ThreadPoolExecutor(max_workers=2) as pool:
        group = _Plans()._submit(pool, 3)
        assert [(name, field, plan) for name, field, plan, _ in group] == _Plans()._plans(3)
        assert [future.result() for *_, future in group] == [plan for *_, plan in _Plans()._plans(3)]
    assert host_submit.submission_failures() == before
