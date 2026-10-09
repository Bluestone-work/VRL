import json
import multiprocessing as mp
from types import SimpleNamespace

import pytest

from marl.evaluation_watchdog import evaluation_results


class Results:
    def __init__(self, items):
        self.items = iter(items)

    def next(self, timeout):
        item = next(self.items)
        if isinstance(item, Exception):
            raise item
        return item


def test_pool_timeout_preserves_completed_cases_and_does_not_fabricate_policy_result(tmp_path):
    complete = dict(anatomy='a', seed=1, method='base', row={'task_success': False})
    pool = SimpleNamespace(imap_unordered=lambda worker, jobs: Results([complete, mp.TimeoutError()]))
    path = tmp_path/'timeout.json'
    results = evaluation_results(pool, None, [('a', 1, 'base'), ('a', 2, 'base')], diagnostic_path=path)
    assert next(results) == complete
    with pytest.raises(RuntimeError, match='1 cases pending'):
        next(results)
    diagnostic = json.loads(path.read_text())
    assert diagnostic['pending'] == [['a', 2, 'base']]
    assert 'task_success' not in diagnostic


def test_empty_pool_returns_without_waiting():
    pool = SimpleNamespace(imap_unordered=lambda worker, jobs: Results([]))
    assert list(evaluation_results(pool, None, [])) == []


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        list(evaluation_results(None, None, [], timeout_s=0.))
