"""The worker pool: ordered results with every start method, and a clear error
when a worker dies instead of a script that waits for ever."""
import multiprocessing
import os
from concurrent.futures.process import BrokenProcessPool

import pytest

from runopts import Workers, n_workers


def square(x):
    return x * x


def die(x):
    if x == 3:
        os._exit(1)          # a worker killed, as by the system running out of memory
    return x


@pytest.mark.parametrize("method", [m for m in ("fork", "spawn") if m in multiprocessing.get_all_start_methods()])
def test_pool_keeps_job_order(method, monkeypatch):
    monkeypatch.setenv("TVABS_START_METHOD", method)
    with Workers(2) as pool:
        assert pool.map(square, range(20)) == [x * x for x in range(20)]


def test_single_worker_runs_in_process():
    calls = []
    with Workers(1, initializer=lambda: calls.append("init")) as pool:
        assert pool.map(square, [1, 2, 3]) == [1, 4, 9]
    assert calls == ["init"]


def test_dead_worker_raises():
    with pytest.raises(BrokenProcessPool):
        with Workers(2) as pool:
            pool.map(die, range(8))


def test_workers_option():
    assert n_workers("doc", ["--workers", "3"]) == 3
    assert n_workers("doc", []) == (os.cpu_count() or 1)
    with pytest.raises(SystemExit):
        n_workers("doc", ["--workers", "0"])
