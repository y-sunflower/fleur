import narwhals as nw
from narwhals.typing import IntoDataFrame


class InputDataHandler:
    def __init__(self, x, y, data: IntoDataFrame = None):
        if isinstance(x, str) and isinstance(y, str):
            if data is None:
                raise ValueError(
                    "If x and y are strings, `data` argument must be passed."
                )
            if x not in data.columns or y not in data.columns:
                raise ValueError("x and/or y not found in data columns.")

            self.data = data
            self.x = data[x]
            self.y = data[y]
            self.x_name = x
            self.y_name = y
            self.source = "dataframe"

        elif isinstance(x, nw.Series) and isinstance(y, nw.Series):
            self.data = None
            self.x = x
            self.y = y
            self.x_name = x.name or "x"
            self.y_name = y.name or "y"
            self.source = "series"

        else:
            raise TypeError(
                "x and y must either be both strings (column names) or both Series."
            )

        self.x_dtype = self.x.dtype
        self.y_dtype = self.y.dtype
        self.x_len = self.x.len()
        self.y_len = self.y.len()

        if self.x_len != self.y_len:
            raise ValueError("x and y must have the same length.")

    def get_info(self):
        return {
            "x_name": self.x_name,
            "y_name": self.y_name,
            "source": self.source,
            "x_dtype": self.x_dtype,
            "y_dtype": self.y_dtype,
            "length": self.x_len,
        }
