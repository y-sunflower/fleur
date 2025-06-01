import narwhals as nw
from narwhals.typing import IntoDataFrame
import os

from typing import Dict, Optional

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
AVAILABLE_DATASETS = ["iris", "mtcars"]
AVAILABLE_OUTPUTS = ["pandas", "polars", "pyarrow", "modin", "cudf"]


def _load_data(dataset_name: str, backend: str, **kwargs) -> IntoDataFrame:
    """
    Load one of the available datasets in fleur. This function is a simple wrapper
    around [`narwhals.read_csv()`](https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.read_csv)
    function.

    Args:
        dataset_name: A string specifying the name of the dataset. Currently, "iris" and "mtcars" are supported.
        backend: The output format of the dataframe. Note that, for example, if you set `backend="polars"`, you must have polars installed. Must be one of the following: "pandas", "polars", "pyarrow", "modin", "cudf". Default to "pandas".
        kwargs: Additional arguments passed to [`narwhals.read_csv()`](https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.read_csv).

    Returns:
        A dataframe with the specified dataset.
    """
    dataset_name = dataset_name.lower()
    backend = backend.lower()

    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"dataset_name must be one of: {' ,'.join(AVAILABLE_DATASETS)}"
        )

    if backend not in AVAILABLE_OUTPUTS:
        raise ValueError(f"backend must be one of: {' ,'.join(AVAILABLE_OUTPUTS)}")

    dataset_file = f"{dataset_name}.csv"
    dataset_path = os.path.join(PACKAGE_DIR, "datasets", dataset_file)
    df = nw.read_csv(dataset_path, backend=backend, **kwargs).to_native()

    return df


def load_iris(backend: str = "pandas", **kwargs: Optional[Dict]) -> IntoDataFrame:
    """
    Load the iris dataset.

    Args:
        backend: The output format of the dataframe. Note that, for example, if you set `backend="polars"`, you must have polars installed. Must be one of the following: "pandas", "polars", "pyarrow", "modin", "cudf". Default to "pandas".
        kwargs: Additional arguments passed to [`narwhals.read_csv()`](https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.read_csv).

    Returns:
        The iris dataset.
    """
    return _load_data("iris", backend=backend, **kwargs)


def load_mtcars(backend: str = "pandas", **kwargs: Optional[Dict]) -> IntoDataFrame:
    """
    Load the mtcars dataset.

    Args:
        backend: The output format of the dataframe. Note that, for example, if you set `backend="polars"`, you must have polars installed. Must be one of the following: "pandas", "polars", "pyarrow", "modin", "cudf". Default to "pandas".
        kwargs: Additional arguments passed to [`narwhals.read_csv()`](https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.read_csv).

    Returns:
        The mtcars dataset.
    """
    return _load_data("mtcars", backend=backend, **kwargs)
