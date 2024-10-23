import time
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn import metrics

try:
    from inference.chisquared_tester import ChiSquaredTester
except ModuleNotFoundError:
    from ..inference.chisquared_tester import ChiSquaredTester


def encode_cols(col):
    try:
        return LabelEncoder().fit_transform(col)
    except TypeError:
        col = col.fillna("")
        return LabelEncoder().fit_transform(col)


class KprototypesGridder:
    def __init__(self, df, sig_variables, verbose):
        self.optimisation_data = []
        self.df = df
        self.sig_vars = sig_variables
        self.chi2_results = []
        self.verbose = verbose  # bool

        # not sure about this part yet
        self.cat_cols_only = [col for col in self.df.columns if
                              col not in list(self.df.select_dtypes('float').columns)]
        self.df_encoded = self.df.apply(encode_cols)

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

    def fluid_all_features_kprototypes_results(self, k, return_chi_2_stat=False):
        if self.verbose:
            print(f"Producing kprototypes results with k == {k} and 'all' features...")
        tic = time.perf_counter()
        catColumnsPos = [self.df_encoded.columns.get_loc(col) for col in self.cat_cols_only]

        dfMatrix = self.df_encoded.to_numpy()

        kprototype = KPrototypes(n_jobs=-1,
                                 n_clusters=k,
                                 init='Cao',
                                 random_state=42,
                                 n_init=10,
                                 verbose=0)
        kmodes = KModes(n_jobs=-1,
                        n_clusters=k,
                        init='Cao',
                        random_state=42,
                        n_init=10,
                        verbose=0)
        kmeans = KMeans(n_clusters=k)
        temp_data = self.df_encoded.copy(deep=True)
        try:
            kprototype.fit_predict(dfMatrix, categorical=catColumnsPos)
            results = {'k': k,
                       'n': 'all',
                       'labels': kprototype.labels_.tolist(),
                       'ch': metrics.calinski_harabasz_score(dfMatrix, kprototype.labels_),
                       'silhouette': metrics.silhouette_score(dfMatrix, kprototype.labels_),
                       'db': metrics.davies_bouldin_score(dfMatrix, kprototype.labels_),
                       'cost': kprototype.cost_,
                       'feats': self.df_encoded.columns.tolist()}
            temp_data['cluster_id'] = kprototype.labels_
        except:
            try:
                print('all data is categorical and we should use kmodes')
                kmodes.fit_predict(dfMatrix)
                results = {'k': k,
                           'n': 'all',
                           'labels': kmodes.labels_.tolist(),
                           'ch': metrics.calinski_harabasz_score(dfMatrix, kmodes.labels_),
                           'silhouette': metrics.silhouette_score(dfMatrix, kmodes.labels_),
                           'db': metrics.davies_bouldin_score(dfMatrix, kmodes.labels_),
                           'cost': kmodes.cost_,
                           'feats': self.df_encoded.columns.tolist()}
                temp_data['cluster_id'] = kmodes.labels_
            except:
                print('try using Kmeans')
                std_data = self.standarize_data(data=temp_data)
                pca_data = self.pca_data(std_data=std_data, find_components=True)
                kmeans.fit_predict(pca_data)
                results = {'k': k,
                           'n': 'all',
                           'labels': kmeans.labels_.tolist(),
                           'ch': metrics.calinski_harabasz_score(dfMatrix, kmeans.labels_),
                           'silhouette': metrics.silhouette_score(dfMatrix, kmeans.labels_),
                           'db': metrics.davies_bouldin_score(dfMatrix, kmeans.labels_),
                           'cost': kmeans.cost_,
                           'feats': self.df_encoded.columns.tolist()}
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
        print(
            f"fluid_all_features_kprototypes_results():\n - the fluid_kproto calculations took {toc - tic:0.4f} seconds for all variables with k={k}")

        if self.verbose:
            print(f"Kprototypes clustering with k={k} and n='all' - COMPLETED")

        return results, sig_vars, chi2_stat

    def n_feats_kprototypes_results(self, k, sig_vars, n, chi2_states):
        if self.verbose:
            print(f"Producing kprototypes results with {n} most significant features for k == {k}...")

        df_feats = self.df_encoded[sig_vars[:n]].copy(deep=True)
        catColPos = [df_feats.columns.get_loc(col) for col in df_feats.columns if col in self.cat_cols_only]
        featMatrix = df_feats.to_numpy()
        n_sig_vars = sig_vars[-n:]
        filtered_chi_2 = chi2_states.set_index('q_code').loc[n_sig_vars]
        if (len(catColPos) != 0) and ((df_feats.shape[1] - len(catColPos)) != 0):
            algo = KPrototypes(n_jobs=-1, n_clusters=k, init='Cao', random_state=42, n_init=10, verbose=0)
            algo.fit_predict(featMatrix, categorical=catColPos)
        elif (len(catColPos) != 0) and ((df_feats.shape[1] - len(catColPos)) == 0):
            if self.verbose:
                print(f"--> Only categorical vars left for k={k} and n={n} - using kmodes ...")
            algo = KModes(n_clusters=k, init='Cao', n_init=10, verbose=0)
            algo.fit_predict(featMatrix)
        else:  # (len(catColPos) == 0) and ((df_feats.shape[1]-len(catColPos)) != 0):

            if self.verbose:
                data = self.df_encoded,
                std_data = self.standarize_data(data=data)
                pca_data = self.pca_data(std_data=std_data, find_components=True)
                algo = KMeans(n_clusters=k)
                algo.fit_predict(pca_data)

        try:
            assertion = algo.labels_ is None
            if assertion:
                if self.verbose:
                    print(f"NOTICE: No results generated for kprototypes clustering with k={k} and n={n}...")
                pass
            else:
                results = {'k': k,
                           'n': n,
                           'labels': algo.labels_.tolist(),
                           'ch': metrics.calinski_harabasz_score(featMatrix, algo.labels_),
                           'silhouette': metrics.silhouette_score(featMatrix, algo.labels_),
                           'db': metrics.davies_bouldin_score(featMatrix, algo.labels_),
                           'cost': algo.cost_,
                           'feats': df_feats.columns.tolist(),
                           'chi2_stat': filtered_chi_2['chi2_stat'].to_list()}

                if self.verbose:
                    print(f"Results generated for kprototypes clustering with k={k} and n={n}...")

                return results
        except:
            print("OH MEOW")
            pass

        if self.verbose:
            print(f"Kprototypes clustering with k={k} and n={n} - COMPLETED")

    def run_fluid_kprototypes_optimisation(self, return_chi_2_stat=False):

        K = range(3, 9)
        n_features = 40
        tic = time.perf_counter()
        results = Parallel(n_jobs=-1)\
            (delayed(self.fluid_all_features_kprototypes_results)(k, return_chi_2_stat=return_chi_2_stat) for k in K)
        # results = [self.fluid_all_features_kprototypes_results(k, return_chi_2_stat=return_chi_2_stat) for k in K]
        toc = time.perf_counter()
        print(
            f"run_fluid_kprototypes_optimisation():\n - the fluid_kproto calculations took {toc - tic:0.4f} seconds for all variables")
        all_vars_results = [res[0] for res in results]
        all_sig_vars_list = [res[1] for res in results]
        all_chi2_stats = [res[2] for res in results]
        self.optimisation_data = self.optimisation_data + all_vars_results

        tic = time.perf_counter()
        for k, sig_vars_list, chi2_stats in zip(K, all_sig_vars_list, all_chi2_stats):
            feats_results = [self.n_feats_kprototypes_results(k, sig_vars_list, n_features, chi2_stats)]

            self.optimisation_data = self.optimisation_data + feats_results
        toc = time.perf_counter()
        print(
            f"run_fluid_kprototypes_optimisation():\n - the fluid_kproto calculations took {toc - tic:0.4f} seconds "
            f"for n-features step")

        return self.optimisation_data
