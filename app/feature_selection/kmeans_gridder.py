import time
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
try:
    from inference.chisquared_tester import ChiSquaredTester
except ModuleNotFoundError:
    from ..inference.chisquared_tester import ChiSquaredTester
import numpy as np


class KMeansGridder:

    def __init__(self, encoded_data, sig_variables, verbose):
        self.optimisation_data = []
        self.df_encoded = encoded_data
        self.sig_vars = sig_variables
        self.verbose = verbose  # bool
        self.chi2_results = []
        self.target_explained_variance = 0.75

    @staticmethod
    def standarize_data(data):
        scaler = StandardScaler()
        try:
            std_data = scaler.fit_transform(data)
        except ValueError:
            print('this is where it is going wrong')
        return std_data

    def find_n_components(self, data):
        pca = PCA(n_components=data.shape[1], svd_solver='auto')
        pca.fit(data)
        cumsums = pca.explained_variance_ratio_.cumsum()
        for i, cumsum in enumerate(cumsums):
            if cumsum < self.target_explained_variance:
                pass
            else:
                ideal_n = i + 1
                break

        return ideal_n

    def pca_data(self, std_data, find_components=True, n_components=None):
        if find_components:
            n = self.find_n_components(data=std_data)
            if self.verbose:
                print(
                    f"- For this PCA {n} components were found to explain {self.target_explained_variance * 100}% of the variance in the dataset")
        else:
            n = n_components
            if self.verbose:
                print(f"- For this PCA {n} components were set by user.")
        pca = PCA(n_components=n, svd_solver='auto')
        pca_data = pca.fit_transform(std_data)
        return pca_data

    def fluid_all_features_kmeans_results(self, k, pca_data, return_chi_2_stat=False):
        """Produces clustering results and significant variables within that clustering output to be provided to fluid
        optimisation run. n PCA components are chosen when threshold cumsum explained variance is met."""
        if self.verbose:
            print(f"Producing fluid kmeans results with k == {k} and 'all' features...")
        tic = time.perf_counter()

        kmeans = KMeans(n_clusters=k)
        kmeans.fit_predict(pca_data)

        results = {'k': k,
                   'n': 'all',
                   'labels': kmeans.labels_.tolist(),
                   'ch': metrics.calinski_harabasz_score(pca_data, kmeans.labels_),
                   'silhouette': metrics.silhouette_score(pca_data, kmeans.labels_),
                   'db': metrics.davies_bouldin_score(pca_data, kmeans.labels_),
                   'feats': self.df_encoded.columns.tolist()}

        temp_data = self.df_encoded.copy(deep=True)
        temp_data['cluster_id'] = kmeans.labels_

        # chi2 part
        tester = ChiSquaredTester(temp_data, 'cluster_id', 0.95, weights=None, correction=None)
        tester.extract_deliver_stats_df(return_chi_2_stat=return_chi_2_stat)
        self.chi2_results = tester.deliver_pg_stats
        self.chi2_results.sort_values(by=['chi2_stat'], inplace=True)
        sig_vars = self.chi2_results.drop_duplicates(subset='q_code', keep='first')['q_code'].tolist()
        chi2_stat = self.chi2_results[['q_code', 'chi_2_result', 'chi2_stat']].drop_duplicates(subset='q_code',
                                                                                               keep='first')
        toc = time.perf_counter()

        if self.verbose:
            print(
                f"fluid_all_features_kmeans_results():\n - the fluid_kmeans calculations took {toc - tic:0.4f} seconds for all variables with k={k}")
            print(f"KMeans clustering with k={k} and n='all' - COMPLETED")

        return results, sig_vars, chi2_stat

    def n_feats_kmeans_results(self, k, sig_vars, n, chi2_stats):
        """Produces clustering results of given k and most n-significant features, PCA component number adjusts to
        the number when threshold cumsum variance is reached"""
        if self.verbose:
            print(f"- Producing kmeans results with {n} most significant features for k == {k}...")
        n_sig_vars = sig_vars[-n:]
        filtered_chi_2 = chi2_stats.set_index('q_code').loc[n_sig_vars]
        std_data = self.standarize_data(data=self.df_encoded[n_sig_vars])
        pca_data = self.pca_data(std_data=std_data)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit_predict(pca_data)

        results = {'k': k,
                   'n': n,
                   'labels': kmeans.labels_.tolist(),
                   'ch': metrics.calinski_harabasz_score(pca_data, kmeans.labels_),
                   'silhouette': metrics.silhouette_score(pca_data, kmeans.labels_),
                   'db': metrics.davies_bouldin_score(pca_data, kmeans.labels_),
                   'feats': n_sig_vars,
                   'chi2_stat': filtered_chi_2['chi2_stat'].to_list()}

        return results

    def all_components_kmeans_results(self, k):
        """produces clustering results with n_components = total number of variables / features in dataset."""
        std_data = self.standarize_data(data=self.df_encoded)
        pca_data = self.pca_data(std_data=std_data,
                                 find_components=False,
                                 n_components=self.df_encoded.shape[1])

        kmeans = KMeans(n_clusters=k)
        kmeans.fit_predict(pca_data)

        results = {'k': k,
                   'n': 'all',
                   'labels': kmeans.labels_.tolist(),
                   'ch': metrics.calinski_harabasz_score(pca_data, kmeans.labels_),
                   'silhouette': metrics.silhouette_score(pca_data, kmeans.labels_),
                   'db': metrics.davies_bouldin_score(pca_data, kmeans.labels_),
                   'feats': self.df_encoded.columns.tolist(),
                   'n_pca_components': self.df_encoded.shape[1]}

        return results

    def n_components_kmeans_results(self, k, n):
        std_data = self.standarize_data(data=self.df_encoded)
        pca_data = self.pca_data(std_data=std_data,
                                 find_components=False,
                                 n_components=n)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit_predict(pca_data)

        results = {'k': k,
                   'n': std_data.shape[1],
                   'labels': kmeans.labels_.tolist(),
                   'ch': metrics.calinski_harabasz_score(pca_data, kmeans.labels_),
                   'silhouette': metrics.silhouette_score(pca_data, kmeans.labels_),
                   'db': metrics.davies_bouldin_score(pca_data, kmeans.labels_),
                   'feats': self.df_encoded.columns.tolist(),
                   'n_pca_components': n}

        return results

    def run_fluid_kmeans_optimisation(self, return_chi_2_stat=False):
        """fluid with pca components at every clustering step"""
        K = range(3, 9)
        n_features = [80]

        # all variables
        if self.verbose:
            print("- Initating standarisation & PCA for encoded dataset...")

        std_data = self.standarize_data(data=self.df_encoded)

        pca_data = self.pca_data(std_data=std_data)
        tic = time.perf_counter()
        results = [self.fluid_all_features_kmeans_results(k, pca_data, return_chi_2_stat) for k in K]
        toc = time.perf_counter()
        if self.verbose:
            print(
                f"run_fluid_kmeans_optimisation():\n - the fluid_kmeans calculations took {toc - tic:0.4f} seconds for all variables")

        all_vars_results = [res[0] for res in results]
        all_sig_vars_list = [res[1] for res in results]
        all_chi2_stats = [res[2] for res in results]
        self.optimisation_data = self.optimisation_data + all_vars_results

        tic = time.perf_counter()

        for k, sig_vars_list, chi2_stats in zip(K, all_sig_vars_list, all_chi2_stats):

            feats_results = Parallel(n_jobs=-1)(
                delayed(self.n_feats_kmeans_results)(k, sig_vars_list, n, chi2_stats) for n in n_features)

            self.optimisation_data = self.optimisation_data + feats_results
        toc = time.perf_counter()
        if self.verbose:
            print(
                f"run_fluid_kmeans_optimisation():\n - the fluid_kmeans calculations took {toc - tic:0.4f} seconds "
                f"for n-features step")

        return self.optimisation_data

    def run_kmeans_components_optimisation(self):
        K = range(3, 8)
        n_components = range(2, 82, 5)

        for k in K:
            result = self.all_components_kmeans_results(k=k)
            self.optimisation_data = self.optimisation_data + [result]

            components_results = Parallel(n_jobs=-2)(delayed(self.n_components_kmeans_results)(k, n) for n in n_components)
            self.optimisation_data = self.optimisation_data + components_results

        return self.optimisation_data
