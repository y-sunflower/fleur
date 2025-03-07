import narwhals as nw
from narwhals.typing import IntoDataFrame
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
AVAILABLE_DATASETS = ["iris"]
AVAILABLE_OUTPUTS = ["pandas", "polars", "pyarrow", "modin", "cudf"]


def load_data(dataset_name: str, return_as: str = "pandas", **kwargs) -> IntoDataFrame:
    """
    Load one of the available datasets in inferplot. This function is a minimalist wrapper around `narwhals.read_csv() <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.read_csv>`_ function.

    :param dataset_name: A string specifying the name of the dataset. Currently, only "iris" is supported.
    :param return_as: The output format of the dataframe. Note that, for example, if you set ``return_as="polars"``, you must have polars installed. Must be one of the following: "pandas", "polars", "pyarrow", "modin", "cudf". Default to "pandas".
    :param kwargs: Additional arguments passed to `narwhals.read_csv() <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.read_csv>`_.
    :return: A dataframe with the specified dataset.

    Examples
    --------

    .. doctest::

        >>> import inferplot.datasets as datasets
        >>> iris = datasets.load_data("iris")
        >>> iris.head()
        sepal_length    sepal_width    petal_length    petal_width    species
                 5.1            3.5             1.4            0.2     setosa
                 4.9            3.0             1.4            0.2     setosa
                 4.7            3.2             1.3            0.2     setosa
                 4.6            3.1             1.5            0.2     setosa
                 5.0            3.6             1.4            0.2     setosa
    """
    dataset_name = dataset_name.lower()
    return_as = return_as.lower()

    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"dataset_name must be one of: {' ,'.join(AVAILABLE_DATASETS)}"
        )

    if return_as not in AVAILABLE_OUTPUTS:
        raise ValueError(f"return_as must be one of: {' ,'.join(AVAILABLE_OUTPUTS)}")

    dataset_file = f"{dataset_name}.csv"
    dataset_path = os.path.join(PACKAGE_DIR, dataset_file)
    df = nw.read_csv(dataset_path, backend=return_as, **kwargs).to_native()

    return df
