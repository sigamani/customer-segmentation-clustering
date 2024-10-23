import pickle

from ..app.metrics import get_significant_variables_and_spread, get_question_group_from_chi2_data

with open('tests/data_for_tests/ChiSquareTester_test_data.pickle', 'rb') as f:
    test_data = pickle.load(f)
chi2_data = test_data['inference_data']


def test_get_question_group_from_chi2_data():
    sig_vars = chi2_data[0][chi2_data[0]['chi_2_result'] <= 0.05]
    result = get_question_group_from_chi2_data(sig_vars, chi2_data[0])
    assert isinstance(result, dict)
    top_level_first_dict_key = list(result.keys())[0]
    assert isinstance(result[top_level_first_dict_key], dict)
    second_level_first_dict_key = list(result[top_level_first_dict_key].keys())[0]
    assert isinstance(result[top_level_first_dict_key][second_level_first_dict_key], float)


def test_get_significant_variables_and_spread():
    result = get_significant_variables_and_spread(chi2_data)
    assert isinstance(result, tuple)
    assert isinstance(result[0], dict)
    assert isinstance(result[0][list(result[0].keys())[0]], int)
    assert len(result[0].keys()) == 5
    assert isinstance(result[1], dict)
    assert isinstance(result[1][list(result[1].keys())[0]], int)
    assert len(result[1].keys()) == 5
    assert isinstance(result[2], dict)
    top_level_first_dict_key = list(result[2].keys())[0]
    assert isinstance(result[2][top_level_first_dict_key], dict)
    second_level_first_dict_key = list(result[2][top_level_first_dict_key].keys())[0]
    assert isinstance(result[2][top_level_first_dict_key][second_level_first_dict_key], float)
    assert isinstance(result[3], dict)
    top_level_first_dict_key = list(result[3].keys())[0]
    assert isinstance(result[3][top_level_first_dict_key], dict)
    second_level_first_dict_key = list(result[3][top_level_first_dict_key].keys())[0]
    assert isinstance(result[3][top_level_first_dict_key][second_level_first_dict_key], float)
