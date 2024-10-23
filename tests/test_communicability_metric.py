import pickle

import pytest

from ..app.metrics import get_communicability

with open('tests/data_for_tests/ChiSquareTester_test_data.pickle', 'rb') as f:
    test_data = pickle.load(f)
chi2_data = test_data['inference_data']


def test_get_communicability():
    communicability_scores = get_communicability(chi2_data)
    assert isinstance(communicability_scores, dict)
    assert all(x in communicability_scores.keys() for x in ['data', 'avg'])
    assert isinstance(communicability_scores['data'], dict)
    assert isinstance(communicability_scores['avg'], float)


def test_get_communicability_wrong_data():
    with pytest.raises(IndexError):
        empty_tup = tuple()
        get_communicability(empty_tup)
