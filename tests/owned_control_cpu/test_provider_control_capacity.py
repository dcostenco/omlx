# SPDX-License-Identifier: Apache-2.0
"""Control-store capacity under the five-minute horizon (oMLX 7ee77d2).

Capacity is one budget that every identity shares: registrations, and the
cancellation tombstones unknown identities leave. Control traffic alone can
fill it; exhaustion refuses (fail closed) and the store heals once the horizon
passes.
"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import uuid

import test_provider_control_store as support

api = support.api
HORIZON_MS = api.RETENTION_SECONDS * 1000
T0 = 1_800_000_000_000  # a fixed wall-clock instant, in milliseconds


def uuid7(ms):
    """A canonical RFC 9562 UUIDv7 minted at `ms`."""
    rand = int.from_bytes(os.urandom(10), "big")
    value = ((ms & ((1 << 48) - 1)) << 80) | (7 << 76) | (((rand >> 62) & 0xFFF) << 64) \
        | (0b10 << 62) | (rand & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-control-capacity-")
        self.addCleanup(self.temp.cleanup)
        self.now = T0
        patcher = mock.patch.object(api, "_wall_ms", lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = api.ControlStore(Path(self.temp.name) / "control", capacity=8)

    def fill_with_unknown_cancellations(self):
        # No device work: each cancellation of an unknown identity leaves a
        # tombstone, and every tombstone counts against the shared budget.
        for _ in range(8):
            self.store.request_cancel(support.OWNER, support.JOB, uuid7(self.now))

    def register(self):
        return self.store.register(**(support.FIELDS | {"operation_id": uuid7(self.now)}))

    def test_control_traffic_alone_exhausts_the_budget_and_registration_refuses(self):
        self.fill_with_unknown_cancellations()
        with self.assertRaises(api.ControlCapacity):
            self.register()

    def test_the_store_heals_once_the_horizon_passes(self):
        self.fill_with_unknown_cancellations()
        self.now += HORIZON_MS - 1
        with self.assertRaises(api.ControlCapacity):
            self.register()
        self.now += 2
        self.assertIsNotNone(self.register())

    def test_the_horizon_is_five_minutes(self):
        # A sustained classifier load (about 1,080 owned operations in 40
        # minutes) was refused at one hour; 1,431 in 680 s ran clean at five.
        self.assertEqual(api.RETENTION_SECONDS, 300)


if __name__ == "__main__":
    unittest.main()
