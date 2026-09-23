"""Using every CPU core, without changing any result.

The text stages of this branch (record building, conclusion masking, duplicate
detection, example building, the leakage scan) are regular expressions over
millions of narratives — CPU work, one document at a time, with no dependency
between documents. They run in worker processes (not threads: Python's lock
would serialise threads on exactly this kind of work).

Two rules every caller keeps:

* **same answer for any worker count** — work is split into chunks whose
  results come back in input order (or are written to parts named by chunk
  index), so ``workers=1`` and ``workers=20`` produce identical output; the
  tests check this for each stage;
* **bounded memory** — at most ``window`` chunks are in flight, so reading a
  multi-gigabyte file never queues the whole file in memory.

The GPU plays no part here; it is used by pretraining and fine-tuning.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing
import os
import time
from collections import deque
from typing import Any, Callable, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

__all__ = ["default_workers", "pmap", "run_windowed"]


def default_workers() -> int:
    """Every core. On the DGX Spark (GB10) that is 20."""
    return max(1, os.cpu_count() or 1)


def _resolve(workers: int | None) -> int:
    return default_workers() if workers is None else max(1, int(workers))


def pmap(function: Callable[[Any], Any], items: Sequence[Any], workers: int | None = None,
         chunksize: int | None = None, minimum: int = 2000) -> list:
    """``[function(x) for x in items]`` across processes, in input order.

    Falls back to a plain loop for one worker or fewer than `minimum` items,
    where starting processes would cost more than it saves.
    """
    workers = _resolve(workers)
    if workers <= 1 or len(items) < minimum:
        return [function(item) for item in items]
    chunksize = chunksize or max(64, len(items) // (workers * 8))
    with multiprocessing.Pool(workers) as pool:
        return list(pool.imap(function, items, chunksize=chunksize))


def run_windowed(function: Callable[[Any], Any], payloads: Iterable[Any], workers: int | None = None,
                 window: int | None = None, initializer: Callable | None = None,
                 initargs: tuple = (), label: str = "chunks",
                 on_result: Callable[[Any], None] | None = None) -> int:
    """Apply `function` to each payload in worker processes, at most `window` in flight.

    Results are handed to `on_result` in SUBMISSION order, so the caller sees
    them exactly as a single-process loop would. Progress is logged every ~30 s
    so a long build is visibly working. Returns the number of payloads.
    """
    workers = _resolve(workers)
    window = window or workers * 2
    started = last_log = time.time()
    done = 0
    payloads = iter(payloads)
    head = list(itertools.islice(payloads, 2))
    if len(head) < 2:                   # one chunk: starting processes would only cost time
        workers = 1
    payloads = itertools.chain(head, payloads)

    def report(force: bool = False) -> None:
        nonlocal last_log
        now = time.time()
        if force or now - last_log >= 30:
            logger.info("%s: %d done in %.0f s (%d processes)", label, done, now - started, workers)
            last_log = now

    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        for payload in payloads:
            result = function(payload)
            if on_result is not None:
                on_result(result)
            done += 1
            report()
        report(force=True)
        return done

    pending: deque = deque()
    with multiprocessing.Pool(workers, initializer=initializer, initargs=initargs) as pool:
        def drain(until: int) -> None:
            nonlocal done
            while len(pending) > until:
                result = pending.popleft().get()
                if on_result is not None:
                    on_result(result)
                done += 1
                report()

        for payload in payloads:
            pending.append(pool.apply_async(function, (payload,)))
            drain(window)
        drain(0)
    report(force=True)
    return done


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
