import json

import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

from ..app.clustering import Clusterings

go_city = pd.read_parquet(
    's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/gocity_brandawareness_germany_q1_2022/gocity_brandawareness_germany_q1_2022_responses/gocity_brandawareness_germany_q1_2022_responses.parquet')
with open('local/data/cluster_cols/go_city.json', 'r') as f:
    go_city_cols = json.load(f)

go_city_cols = [x.lower() for x in go_city_cols]
go_city_cols = [col for col in go_city.columns for x in go_city_cols if x in col]

kmodes_cluster_obj = Clusterings(go_city, go_city_cols)
kmodes_clustering = kmodes_cluster_obj.execute_k_clustering()

vacc_hes = pd.read_parquet(
    's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/aa_vaccinationhesitancy_morocco_q1_2022/aa_vaccinationhesitancy_morocco_q1_2022_responses/aa_vaccinationhesitancy_morocco_q1_2022_responses.parquet')
with open('local/data/cluster_cols/vacc_hes.json', 'r') as f:
    vacc_hes_cols = json.load(f)

vacc_hes_cols = [x.split('_', 1)[1] for x in vacc_hes_cols]
vacc_hes_cols = [x.lower() for x in vacc_hes_cols]
vacc_hes_cols = [col for col in vacc_hes.columns for x in vacc_hes_cols if x in col]
vacc_hes = vacc_hes.loc[:, ~vacc_hes.columns.duplicated()].copy()
vacc_hes_cols = [col for col in vacc_hes.columns if col in vacc_hes_cols]
kprototypes_cluster_obj = Clusterings(vacc_hes, vacc_hes_cols)
kprototypes_clustering = kprototypes_cluster_obj.execute_k_clustering()


def test_execute_k_clustering_kmodes():
    assert len(kmodes_clustering) == 6
    assert isinstance(kmodes_clustering['model'], KModes)
    assert isinstance(kmodes_clustering['labels'], np.ndarray)
    assert isinstance(kmodes_clustering['labels'][0], np.uint16)
    assert isinstance(kmodes_clustering['cluster_centres'], pd.DataFrame)
    assert kmodes_clustering['cluster_centres'].shape[1] == kmodes_cluster_obj.opt_clusters
    assert isinstance(kmodes_clustering['cost'], float)
    assert isinstance(kmodes_clustering['metrics'], dict)
    assert 'calinski_harabasz' in kmodes_clustering['metrics'].keys()


def test_execute_k_clustering_kprototypes():
    assert len(kprototypes_clustering) == 6
    assert isinstance(kprototypes_clustering['model'], KPrototypes)
    assert isinstance(kprototypes_clustering['labels'], np.ndarray)
    assert isinstance(kprototypes_clustering['labels'][0], np.uint16)
    assert isinstance(kprototypes_clustering['cluster_centres'], pd.DataFrame)
    assert kprototypes_clustering['cluster_centres'].shape[1] == kprototypes_cluster_obj.opt_clusters
    assert isinstance(kprototypes_clustering['cost'], float)
    assert isinstance(kprototypes_clustering['metrics'], dict)
    assert 'calinski_harabasz' in kprototypes_clustering['metrics'].keys()


def test_kproto_clustering():
    cluster_df = kprototypes_cluster_obj.data_encoded[kprototypes_cluster_obj.cluster_vars]
    cluster_df = cluster_df.loc[:, ~cluster_df.columns.duplicated()].copy()
    df_matrix = cluster_df.to_numpy()
    cluster_vars = [x for x in cluster_df.columns if x in kprototypes_cluster_obj.cluster_vars]
    cols_cat_index = [cluster_df.columns.get_loc(c) for c in kprototypes_cluster_obj.categorical_cols]
    kproto = kprototypes_cluster_obj.kproto_clustering(kprototypes_cluster_obj.opt_clusters,
                                                       df_matrix, cols_cat_index, cluster_vars, 1, 1)
    assert len(kproto) == 5
    assert isinstance(kproto[0], KPrototypes)
    assert isinstance(kproto[3], np.ndarray)
    assert isinstance(kproto[3][0], np.uint16)
    assert isinstance(kproto[1], pd.DataFrame)
    assert kproto[1].shape[0] == kprototypes_cluster_obj.opt_clusters
    assert isinstance(kproto[2], float)
    assert isinstance(kproto[4], dict)
    assert 'calinski_harabasz' in kproto[4].keys()


def test_kmodes_clustering():
    cluster_df = kmodes_cluster_obj.data_encoded[kmodes_cluster_obj.cluster_vars]
    cluster_df = cluster_df.loc[:, ~cluster_df.columns.duplicated()].copy()
    df_matrix = cluster_df.to_numpy()
    cluster_vars = [x for x in cluster_df.columns if x in kmodes_cluster_obj.cluster_vars]
    kmodes = kmodes_cluster_obj.kmodes_clustering(kmodes_cluster_obj.opt_clusters,
                                                  df_matrix, cluster_vars, 1, 1)
    assert len(kmodes) == 5
    assert isinstance(kmodes[0], KModes)
    assert isinstance(kmodes[3], np.ndarray)
    assert isinstance(kmodes[3][0], np.uint16)
    assert isinstance(kmodes[1], pd.DataFrame)
    assert kmodes[1].shape[0] == kmodes_cluster_obj.opt_clusters
    assert isinstance(kmodes[2], float)
    assert isinstance(kmodes[4], dict)
    assert 'calinski_harabasz' in kmodes[4].keys()
