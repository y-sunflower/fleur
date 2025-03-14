import narwhals as nw
import pandas as pd


def _infer_types(x, y, data):
    """
    Identify which column is categorical and which is numerical in a dataframe.

    Parameters:
        x (str): Name of the first column
        y (str): Name of the second column
        data (pandas.DataFrame): Dataframe containing the columns

    Returns:
        tuple: (categorical_col, numerical_col) - Names of the identified columns

    Raises:
        ValueError: If both columns are categorical or both are numerical
    """

    data = nw.from_native(data).to_pandas()

    if x not in data.columns or y not in data.columns:
        raise ValueError(f"Columns {x} and/or {y} not found in the dataframe")

    def is_categorical(column):
        if isinstance(column, pd.CategoricalDtype) or pd.api.types.is_object_dtype(
            column
        ):
            return True

        if pd.api.types.is_numeric_dtype(column):
            n_unique = column.nunique()
            n_total = len(column)

            if n_unique <= 10 and n_unique / n_total < 0.05:
                return True

        return False

    def is_numerical(column):
        return pd.api.types.is_numeric_dtype(column) and not is_categorical(column)

    x_is_cat = is_categorical(data[x])
    y_is_cat = is_categorical(data[y])

    x_is_num = is_numerical(data[x])
    y_is_num = is_numerical(data[y])

    if x_is_cat and y_is_num:
        return (x, y)
    elif x_is_num and y_is_cat:
        return (y, x)
    else:
        raise ValueError(
            "Either both columns are categorical or both are numerical. "
            "Function requires one categorical and one numerical column."
        )


if __name__ == "__main__":
    data = pd.DataFrame({"y": ["a", "b"], "x": [1, 2]})
    print(_infer_types("x", "y", data))
