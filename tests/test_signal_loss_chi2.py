import numpy as np
import pytest
import pandas as pd
import json
import pickle
from ..app.clustering import Clusterings
from ..app.metrics import get_signal_loss_chi_square

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


def test_chi2_signal_loss_returned_from_clustering():
    metrics = pca_and_kmeans['metrics']
    signal_loss = metrics['chi2_signal_loss']
    for cluster_signal in signal_loss.values():
        try:
            print('blabla')

            assert cluster_signal['f1_score'] >= 0
            assert cluster_signal['f1_score'] <= 1
            assert cluster_signal['signal_loss'] >= 0
            assert cluster_signal['signal_loss'] <= 1
            assert cluster_signal['precision'] >= 0
            assert cluster_signal['precision'] <= 1
            assert cluster_signal['recall_score'] >= 0
            assert cluster_signal['recall_score'] <= 1
        except:
            assert np.isnan(signal_loss)


def test_get_signal_loss():
    signal_loss, _ = get_signal_loss_chi_square(orginal_data=cluster_obj.data, cluster_labels=pca_and_kmeans['labels'],
                                                chi2_data=chi2_data)
    for cluster_signal in signal_loss.values():
        try:
            assert cluster_signal['f1_score'] >= 0
            assert cluster_signal['f1_score'] <= 1
            assert cluster_signal['signal_loss'] >= 0
            assert cluster_signal['signal_loss'] <= 1
            assert cluster_signal['precision'] >= 0
            assert cluster_signal['precision'] <= 1
            assert cluster_signal['recall_score'] >= 0
            assert cluster_signal['recall_score'] <= 1
        except:
            assert np.isnan(signal_loss)


def test_get_signal_wrong_data():
    empty_list = list()
    empty_df = pd.DataFrame()
    empty_tup = tuple()
    signal_loss = get_signal_loss_chi_square(orginal_data=empty_df, cluster_labels=empty_list, chi2_data=empty_tup)
    assert np.isnan(signal_loss)
