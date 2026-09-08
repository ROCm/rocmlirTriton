#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Reader-writer semantics of tuningRunner.py's NumaNodeLock.

Tuning runs many GPUs from one process: benchmark subprocesses share a NUMA
node (shared mode), while a verification run needs the node to itself
(exclusive mode). Getting the lock wrong either serializes the whole run or
lets a verification race the benchmarks it is supposed to check, so the
invariants are pinned here. Pure threading, no GPU involved.

# RUN: %python %s
"""

import os
import shutil
import sys
import threading
import time
import unittest

# tuningRunner.py is deployed next to perfRunner.py under ci-performance-scripts
# and depends on the compiled amd_arch_db binding in that directory.
_script = shutil.which('perfRunner.py')
if _script is None:
    sys.exit("perfRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

from tuningRunner import NumaNodeLock  # noqa: E402


class NumaNodeLockTest(unittest.TestCase):
    """Tests for NumaNodeLock."""

    def test_shared_holders_run_concurrently(self):
        lock = NumaNodeLock()
        num_readers = 4
        entered = threading.Barrier(num_readers + 1, timeout=2.0)
        release = threading.Event()

        def reader():
            lock.acquire_shared()
            try:
                entered.wait()
                release.wait()
            finally:
                lock.release_shared()

        threads = [threading.Thread(target=reader) for _ in range(num_readers)]
        for thread in threads:
            thread.start()
        # Barrier only trips once every reader holds the lock at the same time.
        entered.wait()
        release.set()
        for thread in threads:
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())

    def test_no_overlap_under_contention(self):
        """Stress test: assert all reader/writer exclusion invariants."""
        lock = NumaNodeLock()
        readers_active = 0
        writer_active = False
        state_lock = threading.Lock()
        violations = []
        stop = threading.Event()

        def reader():
            nonlocal readers_active
            while not stop.is_set():
                lock.acquire_shared()
                with state_lock:
                    if writer_active:
                        violations.append("reader saw active writer")
                    readers_active += 1
                with state_lock:
                    readers_active -= 1
                lock.release_shared()

        def writer():
            nonlocal writer_active
            while not stop.is_set():
                lock.acquire_exclusive()
                with state_lock:
                    if readers_active > 0:
                        violations.append("writer saw active readers")
                    if writer_active:
                        violations.append("writer saw another active writer")
                    writer_active = True
                with state_lock:
                    writer_active = False
                lock.release_exclusive()

        threads = ([threading.Thread(target=reader) for _ in range(4)] +
                   [threading.Thread(target=writer) for _ in range(2)])
        for thread in threads:
            thread.start()
        time.sleep(0.5)
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
        self.assertEqual(violations, [], f"Lock invariant violated: {violations}")

    def test_release_shared_without_acquire_is_noop(self):
        """release_shared on a fresh lock must not corrupt state or block a
        subsequent acquire."""
        lock = NumaNodeLock()
        lock.release_shared()
        lock.acquire_exclusive()
        lock.release_exclusive()

    def test_release_exclusive_without_acquire_is_noop(self):
        """release_exclusive on a fresh lock must not corrupt state or block a
        subsequent acquire."""
        lock = NumaNodeLock()
        lock.release_exclusive()
        lock.acquire_shared()
        lock.release_shared()

    def test_release_shared_extra_call_is_noop(self):
        """An extra release_shared after a balanced acquire/release must not push
        the holder count negative."""
        lock = NumaNodeLock()
        lock.acquire_shared()
        lock.release_shared()
        lock.release_shared()
        lock.acquire_exclusive()
        lock.release_exclusive()

    def test_release_exclusive_double_call_is_noop(self):
        """An extra release_exclusive after a balanced acquire/release must not
        flip the flag back."""
        lock = NumaNodeLock()
        lock.acquire_exclusive()
        lock.release_exclusive()
        lock.release_exclusive()
        lock.acquire_shared()
        lock.release_shared()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
