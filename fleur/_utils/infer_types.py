import narwhals as nw


def _infer_types(x, y, df):
    """
    Identify which column is categorical and which is numerical in a dataframe.

    Args
        x Name of the first column
        y Name of the second column
        df A narwhals dataframe containing the columns

    Return (categorical_col, numerical_col) - Names of the identified columns
    """

    def is_categorical(column_name, df):
        col_dtype = df.schema[column_name]

        if isinstance(col_dtype, (nw.Categorical, nw.Enum, nw.String)):
            return True

        return False

    def is_numerical(column_name, df):
        col_dtype = df.schema[column_name]
        return col_dtype.is_numeric() and not is_categorical(column_name, df)

    x_is_cat = is_categorical(x, df)
    y_is_cat = is_categorical(y, df)

    x_is_num = is_numerical(x, df)
    y_is_num = is_numerical(y, df)

    if x_is_cat and y_is_num:
        return (x, y)
    elif x_is_num and y_is_cat:
        return (y, x)
    else:
        raise KeyError(
            "Either both columns are categorical or both are numerical. "
            "Function requires one categorical and one numerical column."
        )
