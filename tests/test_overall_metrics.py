import pytest
import pickle
from ..app.metrics import get_all_metrics


with open('tests/data_for_tests/data_for_test_all_metrics.pickle', 'rb') as f:
    data_tuple = pickle.load(f)


metrics = get_all_metrics(data_tuple[0], data_tuple[1], data_tuple[2], model=data_tuple[3], fitted_model=data_tuple[4],
                          algo=data_tuple[5], chi2_data=data_tuple[6], full_data=data_tuple[7],
                          data_encoded=data_tuple[8])


def test_get_all_metrics():
    assert isinstance(metrics, dict)
    assert len(metrics) > 31
    assert isinstance(metrics['algorithm'], str)
