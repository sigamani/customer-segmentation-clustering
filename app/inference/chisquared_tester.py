import logging
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq

from .helper_functions import remove_not_selected

log = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
log.setLevel(logging.DEBUG)


class ChiSquaredTester:
    """
    Performs a chi2 contingency table statistic on variables among different segments/ clusters.
    It determines if there is a significant difference between the observed frequency and expected frequency counts
    among segments/clusters.

    Parameters
    -----------
    clustered_data: pd.Dataframe
        The response data (processed and cleaned).

    seg_col: str, default: 'cluster'
        The column header which denotes the segment/cluster labels.

    conf_interval: float, default: 0.95
        The confidence interval at which the chi2 should be executed i.e. alpha.
        E.g. 0.95 (95% confidence interval) returns variables as significant with a p-value below 0.05.

    Attributes
    ----------
    cluster_results : dict
        Dictionary of which contains detailed information of each cluster.

    deliver_pg_stats : DataFrame
        Data with necessary variables to populate both Discover and Deliver page elements.

    Notes
    -----
    Documentation of the statistic:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

    """

    def __init__(self, clustered_data, seg_col, conf_interval, weights, correction):
        self.cluster_col = seg_col
        self.ci = conf_interval
        self.weights = weights
        self.correction = correction
        self.data = self.remove_cint(clustered_data)
        self.cluster_results = None
        self.deliver_pg_stats = None

    def remove_cint(self, df):
        df.fillna('not selected', inplace=True)
        cint_cols = [x for x in df.columns if 'cint' in x]
        weight_columns = [x for x in df.columns if 'qudo_weight' in x] # todo: added it to inference_excluded_cols but leaving it for now
        for col in cint_cols + weight_columns:
            try:
                df.drop(col, inplace=True, axis=1)
            except (ValueError, KeyError) as e:
                pass
        return df

    def inference_excluded_cols(self):
        """This allows to exclude from inference but not generally from any other usage cases such as drawing out the
        weighting."""
        cols_to_exclude = []
        if self.weights:
            cols_to_exclude.append(self.weights)

        segmentation_cols = [item for item in self.data.columns if "segmentation" in item.lower()]
        remaining_weights = [item for item in self.data.columns if "weight" in item.lower()]
        cols_to_exclude = cols_to_exclude + segmentation_cols + remaining_weights

        return cols_to_exclude

    def crosstab(self, var):
        """
        Creates the crosstab for a single variable across clusters in frequency counts. It will exclude frequency
        counts of "Not shown".

        Parameters
        -----------
        var: str
            Any variable within the clustered data.

        Returns
        -------
        crosstab_df : pd.DataFrame
            Crosstab with the clusters as columns and the categories within a variable (var) as rows. The cells
            contain the observed frequency counts.
        """

        if "Not shown" in self.data[var].unique().tolist():
            sliced_data = self.data[self.data[var] != "Not shown"].copy(deep=True)
            # todo: this might need to be replaced to still produce a full crosstab but drop columns of 0 entry only
            sliced_data = sliced_data[
                sliced_data[self.cluster_col] != "Not shown"
                ].copy(deep=True)
            # sliced_data.reset_index(inplace=True, drop=True)
            sliced_data = sliced_data.drop(columns=self.inference_excluded_cols())
        else:
            sliced_data = self.data.copy(deep=True)
            sliced_data = sliced_data.drop(columns=self.inference_excluded_cols())

        crosstab_df = pd.crosstab(sliced_data[var], sliced_data[self.cluster_col])
        # todo: think about the zero problem - this is not amended properly yet (variables with crosstab cells below 5
        #  should be dropped
        crosstab_df = crosstab_df.fillna(0.0)
        # crosstab_df = crosstab_df.replace(0, 0.00001)

        for col in crosstab_df.columns:
            crosstab_df = crosstab_df[~(crosstab_df[col] < 5)].copy(deep=True)

        return crosstab_df

    @staticmethod
    def crosstab_percent(crosstab_input):
        """
        Creates the crosstab for a single variable across clusters in percentage within a cluster.

        Note
        ----
        Dependent on crosstab() class function.

        Parameters
        -----------
        crosstab_input: pd.DataFrame
            A crosstab of observed frequencies.

        Returns
        -------
        crosstab_input : pd.DataFrame
            Crosstab with the clusters as columns and the categories within a variable (var) as rows. The cells
            contain the percentage frequency per cluster.
        """

        for k in crosstab_input.columns:
            crosstab_input[k] = round(
                (crosstab_input[k] / crosstab_input[k].sum()) * 100, 1
            )

        return crosstab_input

    @staticmethod
    def expected_crosstab(crosstab_input):
        """
        Creates the crosstab for a single variable across clusters in expected frequency counts.

        Note
        ----
        Dependent on crosstab() class function.

        Parameters
        -----------

        crosstab_input: pd.DataFrame
            A crosstab of observed frequencies.

        Returns
        -------
        crosstab_input : pd.DataFrame
            Crosstab with the clusters as columns and the categories within a variable (var) as rows. The cells
            contain the expected frequency counts per cluster.
        """
        expected = expected_freq(crosstab_input)  # contingency table
        expected = pd.DataFrame(expected)
        expected.index = crosstab_input.index
        expected.columns = crosstab_input.columns

        return expected

    @staticmethod
    def chi2_stats(crosstab_input):
        """
        Does a chi-square test to check if there are significant differences between any of the clusters of a given
        variable (var).


        Note
        ----
        Dependent on crosstab() class function.

        Parameters
        -----------
        crosstab_input: pd.DataFrame
            A crosstab of observed frequencies.

        Returns
        -------
        tuple: a tuple containing:
            - stat (float) : Chi2 statistic
            - p (float) : p-value of the Chi2
            - dof (int) : Degrees of freedom
            - expected (array) : expected frequency count matrix

        """
        try:
            stat, p, dof, expected = chi2_contingency(crosstab_input)
        except ValueError:
            stat, p, dof, expected = None, 1.0, None, None  # todo: think that over

        return stat, p, dof, expected

    @staticmethod
    def direction(crosstab_input, expected_crosstab):
        """
        Indicates if observed values were larger than expected values.

        Note
        ----
        Dependent on crosstab() class function and expected_crosstab() class function.

        Parameters
        -----------
        crosstab_input: pd.DataFrame
            A crosstab.
        expected_crosstab: pd.DataFrame``
            Expected frequencies of crosstab as a crosstable.

        Returns
        -------
        o_minus_e : pd.DataFrame
            Crosstab with the clusters as columns and the categories within a variable (var) as rows. The cells
            contain boolean values: True = observed > expected, else False

        """
        o_minus_e = crosstab_input - expected_crosstab
        o_minus_e = o_minus_e.mask(o_minus_e > 0).isna()

        return o_minus_e

    @staticmethod
    def adjusted_residual(observed_crosstab, expected_crosstab, i, j):
        """
        Calculates the adjusted residual of a value in a contingency table/crosstab. This statistic is used to
        perform a Chi2 post hoc test.

        Ref: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188709

        Parameters
        -----------
        observed_crosstab: pd.DataFrame
            Contingency table with observed frequencies of two variables.

        expected_crosstab: pd.DataFrame
            Contingency table with expected frequencies of two variables.
        i: int
            Row index of desired value.
        j: int
            Column index of desired value.

        Returns
        -------
        adjusted_residual : float
            Adjusted residual of a crosstab value in a specific location [i, j], rounded to 3 decimals.

        """
        row_totals = [item for item in observed_crosstab.sum(axis=1)]
        col_totals = [item for item in observed_crosstab.sum(axis=0)]
        n = sum(row_totals)

        adjusted_residual = (
                (observed_crosstab.iloc[i, j] - expected_crosstab.iloc[i, j])
                / (
                        expected_crosstab.iloc[i, j]
                        * (1 - row_totals[i] / n)
                        * (1 - col_totals[j] / n)
                )
                ** 0.5
        )

        return round(adjusted_residual, 3)

    def chi2_post_hoc_test(self, p_val, crosstab_input, expected_crosstab):
        """
        Performs a chi2 post hoc test i.e. is able to indicate which categories within a significant variable
        (question) are different.

        Parameters
        -----------
        p_val : float
            p-value of a performed chi2_contingency test (scipy) of the inputted crosstable.
        crosstab_input : pd.DataFrame
            Contingency table with observed frequencies of two variables.
        expected_crosstab : pd.DataFrame
            Contingency table with expected frequencies of two variables.

        Returns
        -------
        residuals_df : pd.DataFrame
            Contingency table with test results as pos (significantly above expected value), neg
            (significantly below expected value) or neu (within expected range) in each cell.

        """
        direction = self.direction(
            crosstab_input=crosstab_input.copy(deep=True),
            expected_crosstab=expected_crosstab.copy(deep=True),
        )

        alpha = 1.0 - self.ci

        if p_val <= alpha:

            col_totals = crosstab_input.sum()
            n_cols = len(col_totals)

            row_totals = crosstab_input.sum(axis=1)
            n_rows = len(row_totals)

            residuals_list = []
            p_value_list = []

            for i in range(n_rows):
                residuals = []
                p_values = []

                for j in range(n_cols):

                    adj_res = self.adjusted_residual(
                        crosstab_input, expected_crosstab, i, j
                    )

                    ind_chi_square = adj_res * adj_res
                    p_ind_chi_square = 1 - stats.chi2.cdf(ind_chi_square, 1)

                    z_score_abs = abs(adj_res)

                    if self.correction == 'bonferroni':
                        adjusted_p = alpha / (crosstab_input.shape[0] * crosstab_input.shape[1])
                        p_values.append(adjusted_p)

                        if (z_score_abs >= 1.96) and (p_ind_chi_square <= adjusted_p):
                            if direction.iloc[i, j]:
                                residuals.append("pos")
                            else:
                                residuals.append("neg")
                        else:
                            residuals.append("neu")

                    else:
                        if z_score_abs >= 1.96:
                            if direction.iloc[i, j]:
                                residuals.append("pos")
                            else:
                                residuals.append("neg")
                        else:
                            residuals.append("neu")

                residuals_list.append(residuals)
                p_value_list.append(p_values)

            residuals_df = pd.DataFrame.from_records(
                residuals_list
            )  # df that shows which categories have a significant residual

            residuals_df.columns = crosstab_input.columns
            residuals_df.index = crosstab_input.index

            return residuals_df

        else:
            pass

    def json_cluster_results(self):
        """
        Executes the chi2 contingency table statistic and performs the post hoc test.
        The results are then saved within the class in a dictionary format.

        Returns
        ----------
        results : dict
            json-type dictionary with all clusters and their information collected in the following format:
        e.g. {'cluster 1':
                 {'proportion': 558,
                  'weighted_proportion': None,
                  'percentage_proportion': 37.78,
                  'weighted_percentage_proportion': None,
                  'variables':
                      {'4':
                           {'chi2_stat': 0.0,
                            'p_val': 1.0,
                            'var_significance': False,
                            'TGT': False,
                            'pop_mode': '10003',
                            'pop_mode_perc': 3.58,
                            'weighted_pop_mode': None,
                            'weighted_pop_mode_perc': None,
                            'categories': {},
                            'target': False},
                       '6':
                           {'chi2_stat': 168.7,
                            'p_val': 0.0,
                            'var_significance': True,
                            'TGT': False,
                            'pop_mode': '10008',
                            'pop_mode_perc': 39.25,
                            'weighted_pop_mode': None,
                            'weighted_pop_mode_perc': None,
                            'categories':
                                {'10006':
                                     {'significance': True,
                                      'frequency_count': 51,
                                      'expected_freq_count': 85.54,
                                      'frequency_percentage': 9.5,
                                      'post_hoc_test': 'neg'},
                                 '10007':
                                     {'significance': False,
                                      'frequency_count': 178,
                                      'expected_freq_count': 194.83,
                                      'frequency_percentage': 33.1,
                                      'post_hoc_test': 'neu'},
                                 '10008':
                                     {'significance': True,
                                      'frequency_count': 219,
                                      'expected_freq_count': 177.87,
                                      'frequency_percentage': 40.8,
                                      'post_hoc_test': 'pos'},
                                 '10009': {'significance': False,
                                           'frequency_count': 89,
                                           'expected_freq_count': 78.76,
                                           'frequency_percentage': 16.6,
                                           'post_hoc_test': 'neu'}},
                            'target': True}}},
                            ...}

        """
        variables = self.data.columns.tolist()
        try:
            variables.remove(self.cluster_col)
        except (KeyError, ValueError) as e:
            log.info(
                "json_cluster_results - 2. Could not remove cluster_col from variables - column doesn't exist?"
            )

        to_exclude = self.inference_excluded_cols()
        variables = list(set(variables) - set(to_exclude))

        results = {}

        if not self.data[self.cluster_col].empty:
            cluster_labels = np.sort(
                self.data[self.cluster_col].unique().tolist()
            )  # todo: this throws error in python=3.10 ????

            result_list = Parallel(n_jobs=-1)(
                delayed(self.json_cluster_results_per_segment)(c, variables)
                for c in cluster_labels
            )
            for i in range(len(result_list)):
                for k, v in result_list[i].items():
                    results[k] = v
        else:
            log.info("json_cluster_results - could not detect clusters")

        self.cluster_results = results

        return results

    def calculate_weighted_cluster_proportion(self, cluster):
        sample_total = self.data.shape[0]

        weighted_prop = round(self.data[self.data[self.cluster_col] == cluster][self.weights].sum(), 2)

        weighted_percentage_prop = round(
            (weighted_prop / sample_total) * 100, 2
        )

        return weighted_prop, weighted_percentage_prop

    def calculate_variable_mode(self, cluster, v, proportion):

        mask = (self.data[self.cluster_col] == cluster) & (
                self.data[v] != "Not shown"
        )
        pop_mode = self.data[mask][v].mode().values.tolist()[0]
        pop_mode_perc = round(
            (self.data[mask][v].value_counts()[pop_mode] / proportion)
            * 100,
            2,
        )

        return pop_mode, pop_mode_perc

    def calculate_weighted_variable_mode(self, cluster, v, proportion):

        mask = (self.data[self.cluster_col] == cluster) & (
                self.data[v] != "Not shown"
        )

        category_sizes = self.data[mask].groupby(v)[[self.weights]].sum() / proportion * 100
        category_sizes = category_sizes.reset_index()
        category_sizes.columns = ["category", "value"]

        weighted_pop_mode = category_sizes.max()["category"]
        weighted_pop_mode_perc = round(category_sizes.max()["value"])

        return weighted_pop_mode, weighted_pop_mode_perc

    def append_post_hoc_results(self, p, crosstab, cluster, cat, var):

        expected_crosstab = self.expected_crosstab(
            crosstab_input=crosstab.copy(deep=True)
        )
        percent_crosstab = self.crosstab_percent(
            crosstab_input=crosstab.copy(deep=True)
        )

        post_hoc = self.chi2_post_hoc_test(
            p_val=p,
            crosstab_input=crosstab.copy(deep=True),
            expected_crosstab=expected_crosstab.copy(deep=True),
        )

        if post_hoc.loc[cat, cluster] == "neu":
            sig = False
        else:
            sig = True

        subset = self.data[self.data[self.cluster_col] == cluster].copy(deep=True)
        true_seg_frequency_percentage = dict(subset[var].value_counts(normalize=True)*100)[cat]
        crosstab_frequency_percentage = round(
                percent_crosstab.loc[cat, cluster], 2
            )

        post_hoc_dict = {
            "significance": sig,
            "frequency_count": int(crosstab.loc[cat, cluster]),
            "expected_freq_count": round(
                expected_crosstab.loc[cat, cluster], 2
            ),
            "frequency_percentage": round(true_seg_frequency_percentage, 2),
            "post_hoc_test": post_hoc.loc[cat, cluster],
        }

        if self.weights:
            weighted_frequency_count = subset[subset[var] == cat][self.weights].sum()
            weighted_seg_frequency_percentage = weighted_frequency_count / subset[self.weights].sum() * 100

            post_hoc_dict["weighted_frequency_count"] = weighted_frequency_count
            post_hoc_dict["weighted_frequency_percentage"] = round(weighted_seg_frequency_percentage, 2)

        return post_hoc_dict

    @staticmethod
    def append_target_value(results, cluster, v, categories):

        post_hoc_results = [
            results[cluster]["variables"][v]["categories"][cat][
                "post_hoc_test"
            ]
            for cat in categories
        ]
        if "pos" in post_hoc_results:  # or ('neg' in post_hoc_results):
            # todo: at the moment this only considers more than expected counts
            results[cluster]["variables"][v]["target"] = True
        else:
            results[cluster]["variables"][v]["target"] = False

        return results

    def json_cluster_results_per_segment(self, cluster, variables):
        """

        Parameters
        ----------
        cluster : str, int
            The cluster label for a segment.

        variables :  list

        Returns
        -------
        results : dict
            json-type dictionary with all clusters and their information collected.
        """
        alpha = 1.0 - self.ci
        sample_total = self.data.shape[0]
        proportion = len(self.data[self.data[self.cluster_col] == cluster])
        results = {}

        results[cluster] = {
            "proportion": proportion,
            "percentage_proportion": round((proportion / sample_total) * 100, 2),
            "variables": {},
        }

        if self.weights:
            weighted_prop, weighted_percentage_proportion = self.calculate_weighted_cluster_proportion(cluster=cluster)
            results[cluster]["weighted_proportion"] = weighted_prop
            results[cluster]["weighted_percentage_proportion"] = weighted_percentage_proportion

        variable_count = 0  # todo: this is only used for logger
        tic = time.perf_counter()
        for v in variables:
            variable_count += 1
            crosstab = self.crosstab(var=v)

            stat, p, _, _ = self.chi2_stats(crosstab_input=crosstab.copy(deep=True))

            if stat is None:
                pass
            else:

                if cluster not in crosstab.columns.tolist():
                    pass

                else:
                    pop_mode, pop_mode_perc = self.calculate_variable_mode(cluster=cluster,
                                                                           v=v,
                                                                           proportion=proportion)

                    results[cluster]["variables"][v] = {
                        "chi2_stat": round(stat, 2),
                        "p_val": round(p, 5),
                        "var_significance": True if p <= alpha else False,
                        # 'TGT': True if "_TGT" in v else False, # only activate this if variables are mapped
                        "pop_mode": pop_mode,
                        "pop_mode_perc": pop_mode_perc,
                        "categories": {},
                    }

                    if self.weights:
                        weighted_proportion = self.data[self.data[self.cluster_col] == cluster][self.weights].sum()
                        weighted_pop_mode, weighted_pop_mode_perc = self.calculate_weighted_variable_mode(cluster=cluster,
                                                                                                          v=v,
                                                                                                          proportion=weighted_proportion)
                        results[cluster]["variables"][v]["weighted_pop_mode"] = weighted_pop_mode
                        results[cluster]["variables"][v]["weighted_pop_mode_perc"] = weighted_pop_mode_perc

                    if p <= alpha:

                        categories = crosstab.index.tolist()
                        for cat in categories:
                            results[cluster]["variables"][v]["categories"][cat] = self.append_post_hoc_results(p=p,
                                                                                                               crosstab=crosstab,
                                                                                                               cluster=cluster,
                                                                                                               cat=cat,
                                                                                                               var=v)

                        results = self.append_target_value(results=results,
                                                           cluster=cluster,
                                                           v=v,
                                                           categories=categories)

                    if "target" in results[cluster]["variables"][v]:
                        pass
                    else:
                        results[cluster]["variables"][v]["target"] = False
        toc = time.perf_counter()
        log.info(
            f"json_cluster_results_per_segment - the chi2 stats calculation took {toc - tic:0.4f} seconds for {variable_count} variables "
        )
        if not results:
            log.info("json_cluster_results_per_segment - results empty.")
        return results

    def extract_deliver_stats_df(self, return_chi_2_stat=False) -> bool:
        """
        This function returns a dataframe with all the chi2 significant variables and other relevant information needed
        for the DISCOVER and DELIVER pages' frontend.

        Returns
        -------
        summary_stats_df : pd.DataFrame
            A dataframe with all chi2 contingency table determined significant variables (q_code).

        Notes
        ----
            The returned dataframe contains the following variables:
                q_code:
                    This is the question code that has been given when ingested into the class.
                    Data type: string of data code
                pop_mode:
                    This is the mode answer of the question of q_code within the targeting segment (targeting_seg),
                    NOT within the population.
                    N.B.: This has been named pop_mode because previously this was the mode of the whole dataset NOT
                    within the segment. This may be refactored to "seg_mode".
                    Data type: String of answer code
                    # todo: refactor variable name to seg_mode. Check for frontend compatibility of name and value.
                response_rate:
                    This is the response rate of a question (q_code) within the whole dataset NOT the targeting segment
                    itself. This may be amended to response rate of the segment itself to a certain question (q_code).
                    # todo: change to response rate within segment. Has to be checked for frontend compatibility.
                mode_pop_perc:
                    This is the percentage of the specified targeting segment (targeting_seg) that has the mode value
                    within the variable (q_code).
                    N.B.: This has been named mode_pop_perc because previously this was its name. This was also
                    previously the percentage of the whole dataset containing the mode value for the specified variable
                    (q_code) and NOT just within the targeting segment. This may be refactored to "mode_seg_perc".
                    Data type: Float
                    # todo: refactor variable name to mode_seg_perc. Check for frontend compatibility of name and value.
                chi2_2_result:
                    This is the p-value of a question (q_code) determined from the chi2 contingency table statistic.
                    The lower the p-value the more significant the question answers differed between the segments.
                sig_more_category:
                    These are the values (i.e. categories) within the variable (q_code) that have been significantly
                    "more" different to their expected values and influenced the variable (q_code) to be determined as
                    significant.
                    Data type: List of answers, usually they are codes (strings) depending on what has been ingested.
                targeting_seg:
                    This is the segment (cluster) that the categories (sig_more_category) for each question (q_code)
                    will be targeted on.
                    Data type: String of segment, usually they are codes (strings) depending on what has been ingested.

        """
        deliver_stats_dct = {
            "q_code": [],
            "pop_mode": [],
            "response_rate": [],
            "mode_pop_perc": [],
            "chi_2_result": [],
            "sig_more_category": [],
            "category_percentages": [],
            "targeting_seg": [],
        }
        if return_chi_2_stat:
            deliver_stats_dct['chi2_stat'] = []

        if self.weights:
            deliver_stats_dct["weighted_pop_mode"] = []
            deliver_stats_dct["weighted_pop_mode_perc"] = []
            deliver_stats_dct["weighted_category_percentages"] = []

        tic = time.perf_counter()
        # create results object as json found within class init as self.cluster_results
        self.json_cluster_results()
        toc = time.perf_counter()
        log.info(f"json_cluster_results() finished in {toc - tic:0.4f} seconds")
        segment_ids_found = False
        for segment_id in self.cluster_results.keys():
            segment_ids_found = True
            for v in self.cluster_results[segment_id]["variables"].keys():

                # overall response rate of variable NOT within segment... # todo: discuss this and pop_mode
                response_rate = self.data[v].count() / len(self.data[v]) * 100

                # shortening the json call per variable
                variable_results = self.cluster_results[segment_id]["variables"][v]

                if variable_results["target"]:
                    # only collecting more than expected category counts as features for deliver page
                    pos_cats = []
                    pos_percentages = []
                    weighted_pos_percentages = []
                    for cat in variable_results["categories"]:
                        if (
                                variable_results["categories"][cat]["post_hoc_test"]
                                == "pos"
                        ):
                            pos_cats.append(cat)
                            pos_percentages.append(
                                variable_results["categories"][cat][
                                    "frequency_percentage"
                                ]
                            )
                            if self.weights:
                                weighted_pos_percentages.append(  # todo
                                    variable_results["categories"][cat][
                                        "weighted_frequency_percentage"
                                    ]
                                )

                    deliver_stats_dct["q_code"].append(v)
                    # todo: this used to be the mode of the whole population not the cluster
                    # todo: this might have to be changed back - need to follow-up
                    # todo: was this formula originally: pop_mode = statistics.mode(self.data.loc[:, q_code])
                    deliver_stats_dct["pop_mode"].append(variable_results["pop_mode"])
                    deliver_stats_dct["response_rate"].append(response_rate)
                    deliver_stats_dct["mode_pop_perc"].append(
                        variable_results["pop_mode_perc"]
                    )
                    deliver_stats_dct["chi_2_result"].append(variable_results["p_val"])
                    deliver_stats_dct["sig_more_category"].append(pos_cats)
                    deliver_stats_dct["category_percentages"].append(pos_percentages)
                    deliver_stats_dct["targeting_seg"].append(segment_id)

                    if return_chi_2_stat:
                        deliver_stats_dct['chi2_stat'].append(variable_results['chi2_stat'])

                    if self.weights:
                        deliver_stats_dct["weighted_pop_mode"].append(variable_results["weighted_pop_mode"])
                        deliver_stats_dct["weighted_pop_mode_perc"].append(variable_results["weighted_pop_mode_perc"])
                        deliver_stats_dct["weighted_category_percentages"].append(weighted_pos_percentages)

        if not segment_ids_found:
            return False
        summary_stats_df = pd.DataFrame(deliver_stats_dct).sort_values("chi_2_result")

        # dropping significant results for 'not selected' for sig_more_category variable
        summary_stats_df.sig_more_category = summary_stats_df.sig_more_category.apply(
            remove_not_selected
        )
        # assessing if any significance has been reported
        if summary_stats_df.shape[0] == 0:
            pass  # todo: we need to build in a functionality that flags to the frontend that there is no significance
        else:
            """removing any empty list entries within sig_more_category from dataframe"""
            summary_stats_df = summary_stats_df[
                summary_stats_df["sig_more_category"].str.len() != 0
                ]

        # save to class init
        self.deliver_pg_stats = summary_stats_df

        if summary_stats_df.empty:
            log.info("extract_deliver_stats_df - summary_stats_df empty.")
            return False
        return True

    def seg_discover_stats_df(
            self, seg_name, n_feats=10
    ) -> pd.DataFrame:  # todo: rethink at some point the mode retrieval
        """
        Takes top n features (variables) and returns a dataframe with 'q_code', 'mode', 'mode_perc',
        'sig_more_category' of each variable. This means that one variable can have more than one significant
        category.

        Parameters
        ----------
        seg_name: str
            Segment name (cluster name/label)

        n_feats: int, default: 10
            Number of top features/variables to select

        Returns
        -------
        seg_subset: pd.DataFrame
            A dataframe with segment specific top n_feats (sorted by p-value of chi_2_result), see details in notes.

        Notes
        _____

        A dataframe with segment specific top n_feats (sorted by p-value of chi_2_result) with the following variables:
                q_code:
                    this is the question code that has been given when ingested into the class.
                    Data type: string of data code
                mode:
                    This is the mode answer of the question of q_code within the segment, NOT within the population.
                    Data type: String of answer code
                mode_perc:
                    This is the percentage of the segment that has the mode value with the variable (q_code).
                    Data type: Float
                sig_more_category:
                    These are the values (i.e. categories) within the variable (q_code) that have been significantly
                    "more" different to their expected values and influenced the variable (q_code) to be determined as
                    significant.
                    Data type: List of answers, usually they are codes (strings) depending on what has been ingested.

        """
        # produce a subset of the deliver data per segment
        seg_subset = self.deliver_pg_stats[
            self.deliver_pg_stats.targeting_seg == seg_name
            ].copy(deep=True)
        seg_subset = seg_subset.sort_values(by="chi_2_result", ascending=True)
        seg_subset = seg_subset.rename(
            columns={"pop_mode": "mode", "mode_pop_perc": "mode_perc"}
        )

        seg_subset = seg_subset.drop(
            columns=["response_rate", "chi_2_result", "targeting_seg"]
        )
        # removing any significant variables that have mode "not selected" or "not shown"
        seg_subset = seg_subset[seg_subset["mode"] != "not selected"].copy(deep=True)
        seg_subset = seg_subset[seg_subset["mode"] != "Not shown"].copy(deep=True)

        # logging purposes in sentry if error occurs
        feat_codes = seg_subset[:n_feats]["q_code"]

        return seg_subset[:n_feats]

    def return_API_data(self):
        self.extract_deliver_stats_df()
        deliver_data = self.deliver_pg_stats
        # de-encode for the platform
        deliver_data.response_rate = deliver_data.response_rate.astype("float64")

        if self.cluster_col == "cluster":  # todo: this might be better to be relating to the segmentation_type
            try:
                # answer_dict = {y: x for x, y in iter(self.answers.items())}
                # answer_dict['not selected'] = ""
                # answer_dict['not shown'] = ""
                deliver_data['targeting_seg'] = deliver_data['targeting_seg'].map(int)  # todo: also don't think we need this anymore
                # deliver_data = deliver_data.replace(answer_dict)
                deliver_data['targeting_seg'] = deliver_data['targeting_seg'].map(str)
            except:
                log.error("answers from data agent (--> da.get_answers()) have not been passed to API.")

        discover_data = []
        for seg_name in np.sort(np.unique(self.data[self.cluster_col].tolist())):
            seg_discover_data = self.seg_discover_stats_df(seg_name=str(seg_name))
            discover_data.append({str(seg_name): seg_discover_data})

        mode_list = []
        clusters = self.data.groupby(self.cluster_col)
        for cluster in clusters:
            mode_dict = {str(cluster[0]): cluster[1].mode(dropna=False).head(1)}
            mode_list.append(mode_dict)
        return deliver_data, discover_data, mode_list


if __name__ == "__main__":
    import pandas as pd

    fin_ser = 's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging_responses/qudo_financialservicesfinal_uk_q1_2023_staging_responses.parquet'
    print(fin_ser)
    clustered_data = pd.read_parquet(fin_ser)
    chi2 = ChiSquaredTester(clustered_data=clustered_data, seg_col='qudo', conf_interval=0.95, weights=None,
                            correction=None)

    # output = chi2.return_API_data()
    #
    # meow = pd.DataFrame(output[0])
    #
    # meow[['pop_mode', 'mode_pop_perc', 'sig_more_category', 'category_percentages']].to_csv('test.csv', index=False)
