def _count_n_decimals(f: int | float) -> int:
    """
    Counts the number of decimal places in a floating-point number.

    :param f: The floating-point number whose decimal places are to be counted.
    :return: The number of decimal places in the input number.
    :raises: TypeError
    """

    if not isinstance(f, (float, int)):
        raise TypeError(f"f must be a number, not: {type(f)}")

    s = f"{f}"

    if "." in s:
        integer_part, decimal_part = s.split(".")
        return len(decimal_part.rstrip("0"))
    else:
        return 0
