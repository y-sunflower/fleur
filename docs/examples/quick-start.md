An overview of all things you can do with `fleur`:

## Group comparison

```py
# mkdocs: render
from fleur import BetweenStats, data

df = data.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot()
```

## Correlation

```py
# mkdocs: render
from fleur import ScatterStats, data

df = data.load_iris()

ScatterStats(df["sepal_length"], df["sepal_width"]).plot()
```

## Categorical data comparison

```py
# mkdocs: render
from fleur import BarStats, data

df = data.load_mtcars()
df = df[df["cyl"].isin([4, 6])]

BarStats(df["cyl"], df["vs"]).plot()
```
