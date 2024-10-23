import json
import pickle

import numpy as np
import pandas as pd

from ..app.clustering import Clusterings
from ..app.metrics import get_variability

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


def test_variability_returned_from_clustering():
    metrics = pca_and_kmeans['metrics']
    for cluster_variability in metrics['variability'].values():
        assert cluster_variability >= 0
        assert cluster_variability <= 100


def test_get_variability():
    variability = get_variability(encoded_data=cluster_obj.data_encoded, cluster_labels=pca_and_kmeans['labels'])
    assert isinstance(variability, dict)
    for cluster_variability in variability.values():
        assert cluster_variability >= 0
        assert cluster_variability <= 100


def test_get_variability_wrong_data():
    empty_tup = list()
    empty_df = pd.DataFrame()
    variability = get_variability(encoded_data=empty_df, cluster_labels=empty_tup)
    assert np.isnan(variability)