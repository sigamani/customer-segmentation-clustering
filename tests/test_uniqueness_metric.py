import json
import pickle

import pandas as pd
import pytest

from ..app.clustering import Clusterings
from ..app.metrics import get_uniqueness

part_dem = pd.read_parquet(
    's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/aa_participatorydemocracy_morocco_q4_2022/aa_participatorydemocracy_morocco_q4_2022_responses/aa_participatorydemocracy_morocco_q4_2022_responses.parquet')

with open('local/data/cluster_cols/part_dem.json', 'r') as f:
    part_dem_cols = json.load(f)

part_dem_cols = [x.lower() for x in part_dem_cols]
part_dem_cols = [col for col in part_dem.columns for x in part_dem_cols if x in col]
part_dem = part_dem.loc[:, ~part_dem.columns.duplicated()].copy()
part_dem_cols = [col for col in part_dem.columns if col in part_dem_cols]

cluster_obj = Clusterings(part_dem, part_dem_cols)
pca_and_kmeans = cluster_obj.kmeans_and_pca_clustering()

with open('tests/data_for_tests/ChiSquareTester_test_data.pickle', 'rb') as f:
    test_data = pickle.load(f)
chi2_data = test_data['inference_data']


def test_uniqueness_returned_from_clustering():
    metrics = pca_and_kmeans['metrics']
    for cluster_uniqueness in metrics['uniqueness'].values():
        assert cluster_uniqueness >= 0
        assert cluster_uniqueness <= 1


def test_get_uniqueness():
    uniqueness = get_uniqueness(chi2_data)
    assert isinstance(uniqueness, dict)
    for cluster_uniqueness in uniqueness.values():
        assert cluster_uniqueness >= 0
        assert cluster_uniqueness <= 1


def test_get_uniqueness_wrong_data():
    with pytest.raises(IndexError):
        empty_tup = tuple()
        get_uniqueness(empty_tup)
