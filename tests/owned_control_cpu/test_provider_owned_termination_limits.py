# SPDX-License-Identifier: Apache-2.0
"""27.5 known limit L1, pinned on the exact scheduler methods.

An owned request's untracked deferred cache-store drain still drops the
request when its batch-row removal fails, so the absence probe answers
"absent" although a row may be left. 27.5 records this limit; hardening by
arming the 25a tracker for owned requests is a later slice. Slice 28 closes
it for owned no-retry (Flash-Next) work only: the drain records the failure
and the probe answers "unknown" (test_provider_flash_next_owned).
"""

import ast
from types import SimpleNamespace
import unittest

import test_provider_scheduler_cleanup as cleanup

probe = next(
    n for n in cleanup.scheduler.body
    if isinstance(n, ast.FunctionDef) and n.name == "_probe_owned_presence"
)


class Handle:
    """Stands in for OwnedAdmission, the probe's only type gate."""


ns = {"OwnedAdmission": Handle}
exec(
    compile(
        ast.fix_missing_locations(ast.Module(
            body=[ast.ClassDef(name="Probe", bases=[], keywords=[], body=[probe],
                               decorator_list=[])],
            type_ignores=[],
        )),
        str(cleanup.SOURCE),
        "exec",
    ),
    ns,
)


class DeferredRemoveLimitTests(unittest.TestCase):
    def test_a_failed_untracked_remove_still_reads_absent(self):
        case = cleanup.SchedulerCleanupTests("test_unfinished_store_is_retained_without_cleanup")
        case.setUp()
        s = case.s
        req, future, _ = case.add("owned", 21)
        future.set_result(None)

        def remove(uid):
            if uid == 21:
                raise RuntimeError("synthetic remove failure")
        s._remove_uid_from_active_batch.side_effect = remove
        handle = Handle()
        handle.request = req
        handle.binding = SimpleNamespace(request_id="owned", core=object())
        handle.validate = lambda core, rid: None
        self.assertEqual(ns["Probe"]._probe_owned_presence(s, handle), "finished_before_cleanup")
        with self.assertLogs(level="WARNING"):
            s._drain_pending_async_removes()
        s._remove_uid_from_active_batch.assert_any_call(21)
        # The removal failed, yet the request is gone and the probe proves "absent".
        self.assertNotIn("owned", s.requests)
        self.assertEqual(ns["Probe"]._probe_owned_presence(s, handle), "absent")


if __name__ == "__main__":
    unittest.main()
