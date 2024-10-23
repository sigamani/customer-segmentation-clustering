import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..app.clustering import Clusterings

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


def test_kmeans_and_pca_clustering():
    assert len(pca_and_kmeans) == 6
    assert isinstance(pca_and_kmeans['model'], KMeans)
    assert isinstance(pca_and_kmeans['metrics'], dict)
    assert isinstance(pca_and_kmeans['labels'], np.ndarray)
    assert isinstance(pca_and_kmeans['labels'][0], np.int32)
    assert isinstance(pca_and_kmeans['cost'], float)
    assert isinstance(pca_and_kmeans['cluster_centres'], pd.DataFrame)
    assert len(pca_and_kmeans['labels']) == len(part_dem)


def test_getting_n_components():
    get_optimal_components = cluster_obj.optimal_clusters(method='kmeans_and_pca')
    assert (get_optimal_components, int)


def test_kmeans_and_pca_inertia():
    if cluster_obj.cluster_vars:
        cluster_df = cluster_obj.data_encoded[cluster_obj.cluster_vars]
    else:
        cluster_df = cluster_obj.data_encoded
    cluster_df = cluster_df.loc[:, ~cluster_df.columns.duplicated()].copy()
    vacc_k_p_inertia = cluster_obj.kmeans_and_pca_inertia(3, cluster_df, 1,
                                                          cluster_obj.optimal_clusters(method='kmeans_and_pca'))
    assert len(vacc_k_p_inertia) == 3
    assert isinstance(vacc_k_p_inertia[0], float)
    assert isinstance(vacc_k_p_inertia[1], dict)
    assert isinstance(vacc_k_p_inertia[2], int)
    assert 'calinski_harabasz' in vacc_k_p_inertia[1].keys()


def test_standardize_data():
    std_data = cluster_obj.standarize_data(cluster_obj.data_encoded)
    assert std_data.shape == cluster_obj.data.shape
    assert not pd.DataFrame(std_data).isnull().values.any()
    assert pd.DataFrame(std_data).applymap(lambda x: isinstance(x, (int, float))).values.any()


def test_get_pca_and_kmeans():
    pca_and_kmeans = cluster_obj.get_pca_and_kmeans(cluster_obj.data_encoded[cluster_obj.cluster_vars], 3)
    assert len(pca_and_kmeans) == 3
    assert isinstance(pca_and_kmeans[0], KMeans)
    assert isinstance(pca_and_kmeans[1], dict)
    assert 'calinski_harabasz' in pca_and_kmeans[1].keys()


def test_kmeans_and_pca_no_components():
    kmeans_and_pca = cluster_obj.kmeans_and_pca(cluster_obj.data_encoded[cluster_obj.cluster_vars], 3)
    assert len(kmeans_and_pca) == 3
    assert isinstance(kmeans_and_pca[0], KMeans)
    assert isinstance(kmeans_and_pca[1], int)
    assert isinstance(kmeans_and_pca[2], np.ndarray)


def test_kmeans_and_pca_declared_components():
    number_of_components = 9
    kmeans_and_pca = cluster_obj.kmeans_and_pca(cluster_obj.data_encoded[cluster_obj.cluster_vars], 3,
                                                n_components=number_of_components)
    assert len(kmeans_and_pca) == 3
    assert isinstance(kmeans_and_pca[0], KMeans)
    assert isinstance(kmeans_and_pca[1], int)
    assert kmeans_and_pca[1] == number_of_components
    assert isinstance(kmeans_and_pca[2], np.ndarray)


def get_pca_data_no_components():
    std_data = cluster_obj.standarize_data(cluster_obj.data_encoded[cluster_obj.cluster_vars])
    pca_data = cluster_obj.get_pca_data(std_data)
    assert len(pca_data) == 2
    assert isinstance(pca_data[0], np.ndarray)
    assert isinstance(pca_data[1], int)


def get_pca_data_declared_components():
    number_of_components = 9
    std_data = cluster_obj.standarize_data(cluster_obj.data_encoded[cluster_obj.cluster_vars])
    pca_data = cluster_obj.get_pca_data(std_data, n_components=number_of_components)
    assert len(pca_data) == 2
    assert isinstance(pca_data[0], np.ndarray)
    assert isinstance(pca_data[1], int)
    assert pca_data[1] == number_of_components


def test_get_n_components():
    std_data = cluster_obj.standarize_data(cluster_obj.data_encoded[cluster_obj.cluster_vars])
    n_components = cluster_obj.find_n_components(std_data)
    assert isinstance(n_components, int)
    assert n_components < cluster_obj.data.shape[1]
