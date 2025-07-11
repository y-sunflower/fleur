import narwhals as nw
from narwhals.dependencies import is_numpy_array, is_into_series


class _InputDataHandler:
    def __init__(self, x, y, data=None, backend_for_arrays="pandas"):
        # Case 1: x and y are strings referencing columns from a DataFrame
        if isinstance(x, str) and isinstance(y, str):
            if data is None:
                raise ValueError(
                    "If x and y are strings, `data` argument must be passed."
                )
            if x not in data.columns or y not in data.columns:
                raise ValueError("`x` and/or `y` not found in `data` columns.")

            self.dataframe = nw.from_native(data)
            self.x = self.dataframe[x]
            self.y = self.dataframe[y]
            self.x_name = x
            self.y_name = y
            self.source = "dataframe"

        # Case 2: x and y are Series-like
        elif is_into_series(x) and is_into_series(y):
            self.x = nw.from_native(x, allow_series=True)
            self.y = nw.from_native(y, allow_series=True)
            self.x_name = x.name or "x"
            self.y_name = y.name or "y"
            self.dataframe = nw.from_dict({self.x_name: self.x, self.y_name: self.y})
            self.source = "series"

        # Case 3: x and y are array-like (list, numpy, etc.)
        elif self._is_array_like(x) and self._is_array_like(y):
            self.x = nw.new_series("x", x, backend=backend_for_arrays)
            self.y = nw.new_series("y", y, backend=backend_for_arrays)
            self.x_name = "x"
            self.y_name = "y"
            self.dataframe = nw.from_dict({self.x_name: self.x, self.y_name: self.y})
            self.source = "array"

        else:
            raise TypeError(
                "`x` and `y` must be both strings (column names), "
                "both Series, or both array-like. "
                f"Not type(x)={type(x)} and type(y)={type(y)}"
            )

    def _is_array_like(self, obj):
        return isinstance(obj, (list, tuple)) or is_numpy_array(obj)

    def get_info(self):
        return {
            "x": self.x,
            "y": self.y,
            "x_name": self.x_name,
            "y_name": self.y_name,
            "dataframe": self.dataframe,
            "source": self.source,
        }
