import json

import numpy as np
import pandas as pd

from ..app.clustering import Clusterings

vacc_hes = pd.read_parquet(
    's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/aa_vaccinationhesitancy_morocco_q1_2022/aa_vaccinationhesitancy_morocco_q1_2022_responses/aa_vaccinationhesitancy_morocco_q1_2022_responses.parquet')
with open('local/data/cluster_cols/vacc_hes.json', 'r') as f:
    vacc_hes_cols = json.load(f)

vacc_hes_cols = [x.split('_', 1)[1] for x in vacc_hes_cols]
vacc_hes_cols = [x.lower() for x in vacc_hes_cols]
vacc_hes_cols = [col for col in vacc_hes.columns for x in vacc_hes_cols if x in col]
vacc_hes = vacc_hes.loc[:, ~vacc_hes.columns.duplicated()].copy()
vacc_hes_cols = [col for col in vacc_hes.columns if col in vacc_hes_cols]
vacc_obj = Clusterings(vacc_hes, vacc_hes_cols)
vacc_lca = vacc_obj.lca()

kenzup = pd.read_parquet(
    's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/kenzup_digitalwalletambassadors_morocco_q2_2022/kenzup_digitalwalletambassadors_morocco_q2_2022_responses/kenzup_digitalwalletambassadors_morocco_q2_2022_responses.parquet')

with open('local/data/cluster_cols/kenzup.json', 'r') as f:
    kenz_cols = json.load(f)

kenz_cols = [x.split('_', 1)[1] for x in kenz_cols]
kenz_cols = [x.lower() for x in kenz_cols]
kenz_cols = [col for col in kenzup.columns for x in kenz_cols if x in col]
kenzup = kenzup.loc[:, ~kenzup.columns.duplicated()].copy()
kenz_cols = [col for col in kenzup.columns if col in kenz_cols]

kenz_obj = Clusterings(kenzup, kenz_cols)
kenz_lca = kenz_obj.lca()


def test_lca():
    assert len(kenz_lca) == 4
    assert isinstance(kenz_lca['model'], str)
    assert kenz_lca['model'] == 'poLCA model'
    assert isinstance(kenz_lca['metrics'], dict)
    assert isinstance(kenz_lca['labels'], np.ndarray)
    assert isinstance(kenz_lca['labels'][0], np.int32)
    assert len(kenz_lca['labels']) == len(kenzup)


def test_lca_mixed():
    assert len(vacc_lca) == 4
    assert isinstance(vacc_lca['model'], str)
    assert vacc_lca['model'] == 'poLCA model'
    assert isinstance(vacc_lca['metrics'], dict)
    assert isinstance(vacc_lca['labels'], np.ndarray)
    assert isinstance(vacc_lca['labels'][0], np.int32)
    assert len(vacc_lca['labels']) == len(vacc_hes)


def test_lca_timeout():
    timed_out = vacc_obj.lca(test_wait_time=True)
    assert len(timed_out) == 3
    assert isinstance(timed_out['model'], str)
    assert timed_out['model'] == 'poLCA model'
    assert isinstance(timed_out['metrics'], str)
    assert timed_out['metrics'] == 'lca timed out'
    assert isinstance(timed_out['labels'], str)
    assert timed_out['labels'] == 'lca timed out'


def test_lca_with_timer():
    vacc_lca_with_timer = vacc_obj.lca_with_timer()
    assert len(vacc_lca_with_timer) == 4
    assert isinstance(vacc_lca_with_timer['model'], str)
    assert vacc_lca_with_timer['model'] == 'poLCA model'
    assert isinstance(vacc_lca_with_timer['metrics'], dict)
    assert isinstance(vacc_lca_with_timer['labels'], np.ndarray)
    assert isinstance(vacc_lca_with_timer['labels'][0], np.int32)
    assert len(vacc_lca_with_timer['labels']) == len(vacc_hes)
