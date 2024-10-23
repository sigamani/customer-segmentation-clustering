def remove_not_selected(item: list) -> list:
    """
    Helper function for the ChiSquaredTester class. It removes the string "not selected" from any list.

    Parameters
    ----------
    item : list
        Any list.

    Returns
    -------
    item : list
        The same list but items of "not selected" were removed if present.
    """
    try:
        while True:
            item.remove("not selected")
    except ValueError:
        pass

    return item
