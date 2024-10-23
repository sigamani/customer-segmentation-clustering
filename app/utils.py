def find_weight_col(data, essential_columns) -> str:
    if (not essential_columns["weighting"]["utility"]["pre_completes"]) and (not essential_columns["weighting"]["utility"]["post_completes"]):
        print("Weight column not in dataset - weighting might have not been performed")
        weight_col = None

    elif essential_columns["weighting"]["utility"]["pre_completes"] and essential_columns["weighting"]["utility"]["post_completes"]:
        weight_col = "weight"

    elif essential_columns["weighting"]["utility"]["pre_completes"]:
        weight_col = "precompletion_weight"

    elif essential_columns["weighting"]["utility"]["post_completes"]:
        weight_col = "weight"
    else:
        weight_col = None

    if weight_col.lower() not in data.columns.tolist():
        print("Weight column not in dataset - weighting might have not been performed")
        weight_col = None

    if "weight" in data.columns.tolist() and essential_columns["weighting"]["utility"]["post_completes"]:
        weight_col = "weight"

    return weight_col