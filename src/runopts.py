"""runopts.py -- command-line options and worker processes shared by the
study scripts.

Every script that spreads its runs over worker processes accepts
--workers N; the default is one worker per core. The number of workers
changes the run time only: each run is independent and the results come back
in the order of the jobs.

Workers runs the jobs on a concurrent.futures process pool. A worker that
dies, for example because the system ran out of memory, stops the script with
BrokenProcessPool instead of leaving it waiting. With one worker the jobs run
in the calling process, which needs the least memory. The start method of the
processes is the platform default; the environment variable
TVABS_START_METHOD (fork, spawn or forkserver) overrides it.
"""
import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def parser(doc):
    """Argument parser whose description is the first line of `doc`, with
    the --workers option already added."""
    ap = argparse.ArgumentParser(description=(doc or "").strip().split("\n")[0])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="number of worker processes (default: one per core)")
    return ap


def n_workers(doc, argv=None):
    """Parse the command line of a script that has no option but --workers."""
    n = parser(doc).parse_args(argv).workers
    if n < 1:
        raise SystemExit("--workers must be at least 1")
    return n


class Workers:
    """Process pool with an ordered map, used as a context manager:

        with Workers(4, initializer=init) as pool:
            results = pool.map(fn, jobs)
    """

    def __init__(self, n, initializer=None):
        self.n = max(1, int(n))
        self.initializer = initializer
        self.ex = None

    def __enter__(self):
        if self.n == 1:
            if self.initializer is not None:
                self.initializer()
        else:
            method = os.environ.get("TVABS_START_METHOD")
            ctx = multiprocessing.get_context(method) if method else None
            self.ex = ProcessPoolExecutor(max_workers=self.n, mp_context=ctx, initializer=self.initializer)
        return self

    def map(self, fn, jobs, chunksize=1):
        jobs = list(jobs)
        if self.ex is None:
            return [fn(j) for j in jobs]
        return list(self.ex.map(fn, jobs, chunksize=chunksize))

    def imap_unordered(self, fn, jobs):
        """Yield (index, result) as each job finishes, so a long study can
        save its progress after every run."""
        jobs = list(jobs)
        if self.ex is None:
            for i, j in enumerate(jobs):
                yield i, fn(j)
            return
        futures = {self.ex.submit(fn, j): i for i, j in enumerate(jobs)}
        for f in as_completed(futures):
            yield futures[f], f.result()

    def imap_limited(self, fn, jobs, heavy, max_heavy):
        """As imap_unordered, with at most `max_heavy` of the jobs for which
        heavy(job) is true running at once (to bound the memory); the other
        workers take the remaining jobs meanwhile. Heavy jobs start first."""
        jobs = list(jobs)
        is_heavy = [bool(heavy(j)) for j in jobs]
        queue_h = [i for i in range(len(jobs)) if is_heavy[i]]
        queue_l = [i for i in range(len(jobs)) if not is_heavy[i]]
        if self.ex is None:
            for i in queue_h + queue_l:
                yield i, fn(jobs[i])
            return
        max_heavy = max(1, int(max_heavy))
        running = {}                                      # future -> job index
        while queue_h or queue_l or running:
            while len(running) < self.n:
                n_heavy = sum(is_heavy[i] for i in running.values())
                if queue_h and n_heavy < max_heavy:
                    i = queue_h.pop(0)
                elif queue_l:
                    i = queue_l.pop(0)
                else:
                    break
                running[self.ex.submit(fn, jobs[i])] = i
            done = next(as_completed(running))
            yield running.pop(done), done.result()

    def __exit__(self, *exc):
        if self.ex is not None:
            self.ex.shutdown(wait=True, cancel_futures=exc[0] is not None)
        return False
