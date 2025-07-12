## Plot style

- Default

```python
# mkdocs: render
from fleur import data
from fleur import BetweenStats

import polars as pl

df = (
   data.load_titanic("polars")
   .select(pl.col("Age"), pl.col("Sex"))
   .drop_nulls()
)

BetweenStats(df["Sex"], df["Age"]).plot()
```

<br>

<br>
