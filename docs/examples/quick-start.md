An overview of all things you can do with `fleur`:

## Group comparison

```py
# mkdocs: render
from fleur import BetweenStats, data

df = data.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
   colors=["#005f73", "#ee9b00", "#9b2226"],
   box=False,
)
```

## Correlation

```py
# mkdocs: render
from fleur import ScatterStats, data

df = data.load_iris()

ScatterStats(x=df["sepal_length"], y=df["sepal_width"]).plot(bins=25)
```
