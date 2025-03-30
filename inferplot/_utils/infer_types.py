import narwhals as nw
from narwhals.typing import FrameT


def _infer_types(x, y, data: FrameT):
    """
    Identify which column is categorical and which is numerical in a dataframe.

    :param x: Name of the first column
    :param y: Name of the second column
    :param data: A narwhals dataframe containing the columns
    :return: (categorical_col, numerical_col) - Names of the identified columns
    """

    if x not in data.columns or y not in data.columns:
        raise KeyError(f"Columns {x} and/or {y} not found in the dataframe")

    def is_categorical(column_name):
        col_dtype = data.schema[column_name]

        if isinstance(col_dtype, (nw.Categorical, nw.Enum, nw.String)):
            return True

        if isinstance(col_dtype, nw.Object):
            return True

        return False

    def is_numerical(column_name):
        col_dtype = data.schema[column_name]
        return col_dtype.is_numeric() and not is_categorical(column_name)

    x_is_cat = is_categorical(x)
    y_is_cat = is_categorical(y)

    x_is_num = is_numerical(x)
    y_is_num = is_numerical(y)

    if x_is_cat and y_is_num:
        return (x, y)
    elif x_is_num and y_is_cat:
        return (y, x)
    else:
        raise KeyError(
            "Either both columns are categorical or both are numerical. "
            "Function requires one categorical and one numerical column."
        )
