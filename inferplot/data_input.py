import narwhals as nw
import numpy as np


class InputDataHandler:
    def __init__(self, x, y, data):
        # Case 1: x and y are strings referencing columns from a DataFrame
        if isinstance(x, str) and isinstance(y, str):
            if data is None:
                raise ValueError(
                    "If x and y are strings, `data` argument must be passed."
                )
            if x not in data.columns or y not in data.columns:
                raise ValueError("`x` and/or `y` not found in data columns.")

            self.data = data
            self.x = data[x]
            self.y = data[y]
            self.x_name = x
            self.y_name = y
            self.source = "dataframe"

        # Case 2: x and y are Series --> how to detect that?
        elif isinstance(x, nw.Series) and isinstance(y, nw.Series):
            self.data = None
            self.x = x
            self.y = y
            self.x_name = x.name or "x"
            self.y_name = y.name or "y"
            self.source = "series"

        # Case 3: x and y are array-like (list, numpy, etc.)
        elif self._is_array_like(x) and self._is_array_like(y):
            self.data = None
            self.x = nw.new_series(x)
            self.y = nw.new_series(y)
            self.x_name = "x"
            self.y_name = "y"
            self.source = "array"

        else:
            raise TypeError(
                "`x` and `y` must be both strings (column names), both Series, or both array-like."
            )

        if self.x.len() != self.y.len():
            raise ValueError("`x` and `y` must have the same length.")

    def _is_array_like(self, obj):
        return isinstance(obj, (list, tuple, np.ndarray))

    def get_info(self):
        return {
            "x": self.x,
            "y": self.y,
            "x_name": self.x_name,
            "y_name": self.y_name,
            "source": self.source,
        }
