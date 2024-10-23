import numpy as np
import pandas as pd
import boto3
import json

import pytest

from ..app.clustering import Clusterings
from ..app.main import remove_na_strings


us_fin_ser = 's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_financialservices_usa_q1_2023_staging/qudo_financialservices_usa_q1_2023_staging.parquet'

fin_ser = pd.read_parquet(us_fin_ser)
fin_ser = fin_ser.apply(remove_na_strings)
new_cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in fin_ser.columns]
fin_ser.columns = new_cols

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

columns_uri = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservices_usa_q1_2023_staging/2023-03-07_16:19:20/cols.json'
splits = columns_uri.split('/')
bucket_name = splits[2]
key_name = "/".join(splits[3:])
content_obj = s3.Object(bucket_name, key_name)
content = content_obj.get()['Body'].read().decode('utf-8')
fin_ser_cols = json.loads(content)
fin_ser_cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in fin_ser_cols]

gender_col = [x for x in fin_ser.columns if 'gender' in x][0]

cluster_obj = Clusterings(fin_ser, fin_ser_cols, hierarchical=gender_col)

def test_instantiation():
    assert isinstance(cluster_obj, Clusterings)


def test_hierarchical_kmeans_and_pca():
    hier_kmeans = cluster_obj.hierarchical_kmeans_and_pca()
    assert isinstance(hier_kmeans, dict)
    assert isinstance(hier_kmeans['model'], str)
    assert isinstance(hier_kmeans['labels'], pd.Series)
    assert isinstance(hier_kmeans['metrics'], dict)
    assert isinstance(hier_kmeans['inference_data'], tuple)


def test_hierarchical_lca():
    hier_lca = cluster_obj.hierarchical_lca()
    assert isinstance(hier_lca, dict)
    assert isinstance(hier_lca['model'], str)
    assert isinstance(hier_lca['labels'], pd.Series)
    assert isinstance(hier_lca['metrics'], dict)
    assert isinstance(hier_lca['inference_data'], tuple)


def test_hierarchical_bmm():
    hier_bmm = cluster_obj.hierarchical_bmm()
    assert isinstance(hier_bmm, dict)
    assert isinstance(hier_bmm['model'], str)
    assert isinstance(hier_bmm['labels'], pd.Series)
    assert isinstance(hier_bmm['metrics'], dict)
    assert isinstance(hier_bmm['inference_data'], tuple)


def test_hierarchical_k_clustering():
    hier_kproto = cluster_obj.hierarchical_k_clustering()
    assert isinstance(hier_kproto, dict)
    assert isinstance(hier_kproto['model'], str)
    assert isinstance(hier_kproto['labels'], pd.Series)
    assert isinstance(hier_kproto['metrics'], dict)
    assert isinstance(hier_kproto['inference_data'], tuple)


cluster_obj_ignore_value = Clusterings(fin_ser, fin_ser_cols, hierarchical=gender_col, ignore_hierarchical_value='Male')


def test_ignored_instantiation():
    assert isinstance(cluster_obj_ignore_value, Clusterings)


def test_ignored_hierarchical_kmeans_and_pca():
    hier_kmeans = cluster_obj_ignore_value.hierarchical_kmeans_and_pca()
    assert isinstance(hier_kmeans, dict)
    assert isinstance(hier_kmeans['model'], str)
    assert isinstance(hier_kmeans['labels'], pd.Series)
    assert isinstance(hier_kmeans['metrics'], dict)
    assert isinstance(hier_kmeans['inference_data'], tuple)


def test_ignored_hierarchical_lca():
    hier_lca = cluster_obj_ignore_value.hierarchical_lca()
    assert isinstance(hier_lca, dict)
    assert isinstance(hier_lca['model'], str)
    assert isinstance(hier_lca['labels'], pd.Series)
    assert isinstance(hier_lca['metrics'], dict)
    assert isinstance(hier_lca['inference_data'], tuple)


def test_ignored_hierarchical_bmm():
    hier_bmm = cluster_obj_ignore_value.hierarchical_bmm()
    assert isinstance(hier_bmm, dict)
    assert isinstance(hier_bmm['model'], str)
    assert isinstance(hier_bmm['labels'], pd.Series)
    assert isinstance(hier_bmm['metrics'], dict)
    assert isinstance(hier_bmm['inference_data'], tuple)


def test_ignored_hierarchical_k_clustering():
    hier_kproto = cluster_obj_ignore_value.hierarchical_k_clustering()
    assert isinstance(hier_kproto, dict)
    assert isinstance(hier_kproto['model'], str)
    assert isinstance(hier_kproto['labels'], pd.Series)
    assert isinstance(hier_kproto['metrics'], dict)
    assert isinstance(hier_kproto['inference_data'], tuple)

