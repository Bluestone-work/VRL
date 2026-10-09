import json
from scripts.multicluster_protocol import reserved_seed


def test_exp0062_access_offset_is_development_and_other_multicluster_test_is_reserved():
    assert reserved_seed(2700000011) is False
    assert reserved_seed(2700000012) is True
