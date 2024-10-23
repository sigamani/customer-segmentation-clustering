import pandas as pd
import pytest

try:
    from app.feature_selection.feature_selection import rank_features_laplacian, get_optimal_laplacian, \
        laplacian_score_quality
except ModuleNotFoundError:
    from ..app.feature_selection.feature_selection import rank_features_laplacian, get_optimal_laplacian, \
        laplacian_score_quality

fin_ser = 's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging_responses/'

rdf = pd.read_parquet(fin_ser)
sbeh_cols = [x for x in rdf.columns if any(specific_col in x for specific_col in ['sbeh', 'aida'])]


def test_rank_features_laplacian_output():
    test_df = rdf.copy()
    df_ranked_features = rank_features_laplacian(test_df)
    assert isinstance(df_ranked_features, pd.DataFrame)
    assert len(df_ranked_features.columns) == 40


def test_rank_features_laplacian_no_sbeh():
    test_df = rdf.drop(columns=sbeh_cols).copy()
    with pytest.raises(ValueError, match=r"No SBEH columns present."):
        rank_features_laplacian(test_df)


def test_rank_features_laplacian_wrong_input():
    test_df = dict(rdf)
    with pytest.raises(TypeError, match=r"Input must be a DataFrame."):
        rank_features_laplacian(test_df)


def test_optimal_laplacian():
    test_df = rdf.copy()
    optimal_laplacian_dict = get_optimal_laplacian(test_df)
    assert isinstance(optimal_laplacian_dict, dict)
    assert len(optimal_laplacian_dict) == 12
    assert isinstance(optimal_laplacian_dict['n_neighbours'], int)
    assert isinstance(optimal_laplacian_dict['n_clusters'], int)
    assert isinstance(optimal_laplacian_dict['cols'], list)
    assert len(optimal_laplacian_dict['cols']) == 40
    assert isinstance(optimal_laplacian_dict['rank_sum'], float)


def test_optimal_laplacian_wrong_neighbours():
    test_df = rdf.copy()
    with pytest.raises(ValueError, match=r"min_neighbours must be less than or equal to max_neighbours."):
        get_optimal_laplacian(test_df, min_neighbours=30, max_neighbours=25)


def test_optimal_laplacian_wrong_input():
    test_df = dict(rdf)
    with pytest.raises(TypeError, match=r"Input must be a DataFrame."):
        get_optimal_laplacian(test_df)


def test_laplacian_score_quality():
    test_df = rdf.copy()
    laplacian_score_quality_dict = laplacian_score_quality(test_df)
    assert (laplacian_score_quality_dict, dict)
    assert len(laplacian_score_quality_dict) == 8
    assert isinstance(laplacian_score_quality_dict['n_neighbours'], int)
    assert isinstance(laplacian_score_quality_dict['n_clusters'], int)
    assert isinstance(laplacian_score_quality_dict['cols'], list)
    assert len(laplacian_score_quality_dict['cols']) == 40


def test_laplacian_score_quality_wrong_neighbours_input():
    test_df = rdf.copy()
    with pytest.raises(TypeError, match=r"n_neighbours must be a positive integer."):
        laplacian_score_quality(test_df, n_neighbours='five')
