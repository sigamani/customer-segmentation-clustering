import numpy as np
import pytest
import pandas as pd
import json
import pickle
from ..app.clustering import Clusterings
from ..app.metrics import get_message_reach_metric

path = 's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/aa_participatorydemocracy_morocco_q4_2022/aa_participatorydemocracy_morocco_q4_2022_responses/aa_participatorydemocracy_morocco_q4_2022_responses.parquet'
part_dem = pd.read_parquet(path)
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


def test_message_reach_returned_from_clustering():
    metrics = pca_and_kmeans['metrics']
    for cluster_message_reach in metrics['message_reach_ml_signal'].values():
        try:
            assert cluster_message_reach >= 0
            assert cluster_message_reach <= 1
        except:
            assert np.isnan(cluster_message_reach)
    for cluster_message_reach in metrics['massage_reach_chi2_signal'].values():
        try:
            assert cluster_message_reach >= 0
            assert cluster_message_reach <= 1
        except:
            assert np.isnan(cluster_message_reach)

    for cluster_message_reach in metrics['message_reach_optimal_signal'].values():
        try:
            assert cluster_message_reach >= 0
            assert cluster_message_reach <= 1
        except:
            assert np.isnan(cluster_message_reach)


def test_get_message_reach():
    metrics = pca_and_kmeans['metrics']
    ml_signal = metrics['ml_signal']
    chi2_signal = metrics['chi2_signal']
    optimal_signal = metrics['chi2_signal_core_columns']
    presence = metrics['fb_presence']
    msg_ml_signal = get_message_reach_metric(social_presence=presence, signal_loss=ml_signal)
    msg_chi_signal = get_message_reach_metric(social_presence=presence, signal_loss=chi2_signal)
    msg_opt_signal = get_message_reach_metric(social_presence=presence, signal_loss=optimal_signal)
    for (msg_ml, msg_chi, msg_opt) in zip(msg_ml_signal.values(), msg_chi_signal.values(), msg_opt_signal.values()):
        try:
            assert msg_ml >= 0
            assert msg_ml <= 1
            assert msg_chi >= 0
            assert msg_chi <= 1
            assert msg_opt >= 0
            assert msg_opt <= 1
        except:
            assert np.isnan(msg_ml_signal)
            assert np.isnan(msg_chi_signal)
            assert np.isnan(msg_opt_signal)


def test_get_message_reach_wrong_data():
    empty_dict = dict()
    message_reach = get_message_reach_metric(social_presence=empty_dict, signal_loss=empty_dict)
    assert np.isnan(message_reach)