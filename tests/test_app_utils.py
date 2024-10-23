import json
import pandas as pd
from ..app.utils import find_weight_col

part_dem = pd.read_parquet('s3://qudo-datascience/data-store/surveys/staging/responses_cleaned/aa_participatorydemocracy_morocco_q4_2022_staging/aa_participatorydemocracy_morocco_q4_2022_staging.parquet')

with open('tests/data_for_tests/essentialcolumns.json', 'r') as file:
    essential_columns = json.load(file)


def test_find_weight_col():
    weight_col = find_weight_col(data=part_dem,
                                 essential_columns=essential_columns)

    assert isinstance(weight_col, str)
    assert weight_col == "weight"