"""Bound evaluation-pool silence without labelling an infrastructure hang a policy failure."""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path


def evaluation_results(pool, worker, jobs, timeout_s=120., diagnostic_path=None):
    if timeout_s <= 0:
        raise ValueError('Evaluation timeout must be positive')
    iterator = pool.imap_unordered(worker, jobs)
    received = set()
    for _ in range(len(jobs)):
        try:
            result = iterator.next(timeout=timeout_s)
        except mp.TimeoutError as error:
            pending = [list(job[:3]) for job in jobs if tuple(job[:3]) not in received]
            diagnostic = dict(infrastructure_timeout_s=float(timeout_s), completed=len(received), pending=pending,
                              interpretation='No policy outcome assigned; resume identical unfinished cases')
            if diagnostic_path:
                Path(diagnostic_path).write_text(json.dumps(diagnostic, indent=2)+'\n')
            raise RuntimeError(f'Evaluation pool produced no result for {timeout_s:g} s; {len(pending)} cases pending') from error
        received.add((result['anatomy'], result['seed'], result['method']))
        yield result
