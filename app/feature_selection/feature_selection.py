import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from .kmeans_gridder import KMeansGridder
from .kprototypes_gridder import KprototypesGridder
from .laplacian_score import LaplacianScore, feature_ranking
from .multi_cluster_feature_selection import mcfs, mc_feature_ranking
from .utils import construct_W

try:
    from ..clustering import Clusterings
except:
    from clustering import Clusterings


def get_laplacian_features(df, n_features, n_neighbours, t, return_ranking=False):
    lap_score = LaplacianScore(df, metric='gower', neighbour_size=n_neighbours, t_param=t)
    ranked_features = feature_ranking(lap_score)
    column_ranking = pd.DataFrame(df.columns, columns=['feature'])
    column_ranking['rank'] = ranked_features

    idx = np.argpartition(ranked_features, n_features)
    df_feature_selected = df.iloc[:, ranked_features[idx[-n_features:]]]
    if return_ranking:
        return df_feature_selected, column_ranking.sort_values(by='rank', ascending=False)
    return df_feature_selected


def laplacian_score_quality(df, n_neighbours=15, n_cols=40):
    '''Function aims to score the laplacian score feature selection by running a simple kmeans_and_pca_clustering
    on the selected columns and computing the clustering quality metrics'''
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a DataFrame.')
    if not isinstance(n_neighbours, int):
        raise TypeError('n_neighbours must be a positive integer.')
    if n_neighbours <= 0:
        raise ValueError('n_neighbours must be a positive integer.')

    quality_dict = {'n_neighbours': n_neighbours, 'n_cols': n_cols}
    start_time = time.time()
    lap_features, col_ranking = get_laplacian_features(df, n_cols, n_neighbours, 1, return_ranking=True)
    cols = list(lap_features.columns)
    segmentation_obj = Clusterings(df, cols)
    model = segmentation_obj.kmeans_and_pca_clustering(mode='tune')
    run_time = time.time()
    metrics = model['metrics']
    quality_dict['n_clusters'] = metrics['n_clusters']
    quality_dict['silhouette'] = metrics['silhouette']
    quality_dict['davies_bouldin'] = metrics['davies_bouldin']
    quality_dict['calinski_harabasz'] = metrics['calinski_harabasz']
    quality_dict['time_taken'] = run_time - start_time
    quality_dict['cols'] = cols
    quality_dict['column_rankings'] = col_ranking
    return quality_dict


def encode_cols(col):
    try:
        return LabelEncoder().fit_transform(col)
    except TypeError:
        col = col.fillna("")
        return LabelEncoder().fit_transform(col)


def get_optimal_laplacian(df, min_neighbours=5, max_neighbours=30, n_cols=40):
    '''Function aims to run multiple instances of Laplacian Score feature selection with different n_neighbours values.
    Once we obtain the clustering quality of each instance, we can then select the optimal instance and output the resulting selected columns'''
    if min_neighbours > max_neighbours:
        raise ValueError('min_neighbours must be less than or equal to max_neighbours.')

    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a DataFrame.')

    # # unhash if Parallel works for you
    inputs = tqdm(list(range(min_neighbours, max_neighbours + 5, 5)))
    parallel_processed = Parallel(n_jobs=-1)(
        delayed(laplacian_score_quality)(df, n_neighbours=i, n_cols=n_cols)
        for i in inputs
    )

    # # since i cannot run parallel on my linux machine
    # inputs = range(min_neighbours, max_neighbours+5, 5)
    # parallel_processed = [laplacian_score_quality(df, n_neighbours=i, n_cols=n_cols) for i in inputs]

    evaluation_df = pd.DataFrame(parallel_processed)
    evaluation_df['silhouette_rank'] = evaluation_df['silhouette'].rank()
    evaluation_df['davies_bouldin_rank'] = evaluation_df['davies_bouldin'].rank(ascending=False)
    evaluation_df['calinski_harabasz_rank'] = evaluation_df['calinski_harabasz'].rank()
    rank_cols = ['silhouette_rank', 'davies_bouldin_rank', 'calinski_harabasz_rank']
    evaluation_df['rank_sum'] = evaluation_df[rank_cols].sum(axis=1)
    best_clustering_index = evaluation_df['rank_sum'].idxmax()
    best_results = evaluation_df.iloc[best_clustering_index].to_dict()
    return best_results
    # return df[best_results['cols']]


def get_multi_cluster(df, n_features, n_neighbours, t, n_clusters, cluster_type='kproto', return_ranking=False):
    if cluster_type == 'kproto':
        w = construct_W(df, metric='gower', neighbour_size=n_neighbours, t_param=t)
        multi_cluster = mcfs(df.to_numpy(), n_features, W=w, n_clusters=n_clusters, metric='gower')
    else:
        multi_cluster = mcfs(df.to_numpy(), n_features, n_clusters=n_clusters)
    ranked_features, weights = mc_feature_ranking(multi_cluster)
    if return_ranking:
        return ranked_features, weights
    idx = np.argpartition(ranked_features, n_features)
    df_feature_selected = df.iloc[:, ranked_features[idx[-n_features:]]]
    return df_feature_selected, weights


def get_chi_squared_ranking(df, cluster_type='kproto'):
    if df.shape[0] == 0:
        return np.nan

    if cluster_type == 'kproto' and all(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())):
        kproto_gridder = KprototypesGridder(df=df,
                                            sig_variables=None,
                                            verbose=True)
        try:
            optimisation_data = kproto_gridder.run_fluid_kprototypes_optimisation(return_chi_2_stat=True)
        except:
            return np.nan

    elif cluster_type == 'kmeans':
        kmeans_gridder = KMeansGridder(encoded_data=df,
                                       sig_variables=None,
                                       verbose=True)
        try:
            optimisation_data = kmeans_gridder.run_fluid_kmeans_optimisation(return_chi_2_stat=True)
        except:
            return np.nan
    else:
        return np.nan

    for dat in optimisation_data:
        if dat['n'] != 'all':
            dat_df = pd.DataFrame({'q_code': dat['feats'], f'stat - {dat["k"]}': dat['chi2_stat']})
            try:
                all_sig_vars = pd.merge(all_sig_vars, dat_df, on='q_code', how='outer')
            except NameError:
                all_sig_vars = dat_df

    all_sig_vars.set_index('q_code', inplace=True)
    ranking = all_sig_vars.apply(lambda x: x.rank(ascending=False, na_option='bottom'))
    ranking['total'] = ranking.sum(axis=1)
    results = ranking['total'].rank()
    results.sort_values(inplace=True)
    return results


def get_column_names(df):
    return [column for column in df.columns]


def sort_feature_cols(varnames, rank):
    return [x for _, x in sorted(zip(rank, varnames))]


def rank_features(df, n_features, n_neighbours, t, n_clusters):
    """Only consider study-specific questions (UK/US Financial, US GoCity Brand Awareness)"""
    if not isinstance(n_features, int):
        raise TypeError('n_features must be positive Integer')
    if n_features <= 0:
        raise ValueError('n_features must be positive Integer')
    if not isinstance(df, pd.DataFrame):
        raise TypeError('input should be pandas DataFrame')
    if not isinstance(n_neighbours, int):
        raise TypeError('n_neighbours must be a positive Integer.')
    if n_neighbours <= 0:
        raise ValueError('n_neighbours must be a positive Integer.')
    if n_clusters <= 0:
        raise ValueError('n_clusters must be positive Integer')
    if not isinstance(n_clusters, int):
        raise TypeError('n_clusters must be positive Integer')
    if n_features > len(df):
        raise ValueError('n_features is more than number of columns')
    sbeh_cols = [x for x in df.columns if any(specific_col in x for specific_col in ['sbeh', 'aida'])]
    df = df[sbeh_cols]
    df_encoded = df.apply(LabelEncoder().fit_transform)

    """Chi-squared feature ranking"""
    chi_squared_ranking_kmeans_pca = get_chi_squared_ranking(df_encoded, cluster_type='kmeans').head(n_features)
    chi_squared_ranking_kproto = get_chi_squared_ranking(df_encoded).head(n_features)
    """Multi-cluster feature ranking. Weights are calculated if we want to use them."""
    mcfs_score, weights = get_multi_cluster(df, n_features, n_neighbours, t, n_clusters, return_ranking=True)
    mcfs_score += 1
    mcsf = pd.DataFrame(mcfs_score).set_index(df.columns)
    mcsf_result = mcsf.sort_values(by=0).head(n_features)
    """Laplacian feature ranking"""
    laplacian_results = get_optimal_laplacian(df)
    lap_df = laplacian_results['column_rankings']
    lap_df['rank'] = lap_df['rank'].rank(ascending=False)
    lap_ranks = lap_df.head(n_features).set_index('feature')
    series = [chi_squared_ranking_kmeans_pca, chi_squared_ranking_kproto, mcsf_result, lap_ranks]
    result = pd.concat(series, axis=1).reset_index()
    result.columns = ['index', 'Chi2_Kmeans', 'Chi2_kproto', 'MCSF', 'Laplacian']
    result.fillna(n_features * 1.1, inplace=True)
    result['sum_rank'] = result.sum(axis=1)
    # ToDo: Write a function to save this out to s3 so we have an audit trail
    result.sort_values(by='sum_rank', inplace=True)
    final_cols = result.head(n_features)

    """
    In the paper they suggest using the normalized mutual information metric (NMI) 
    with the ground truth labels to caluclate cluster quality:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html 
    """

    return final_cols['index'].to_list()


def rank_features_laplacian(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a DataFrame.')

    sbeh_cols = [x for x in df.columns if any(specific_col in x for specific_col in ['sbeh', 'aida'])]
    if not sbeh_cols:
        raise ValueError('No SBEH columns present.')
    df = df[sbeh_cols]
    laplacian_score = get_optimal_laplacian(df)
    laplacian_df = df[laplacian_score['cols']]
    return laplacian_df
