A bunch of examples of things you can do with `fleur`:

<br>

- `BetweenStats()`

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot()
```

<br>

- `ScatterStats()`

```python
# mkdocs: render
from fleur import ScatterStats
from fleur import datasets

df = datasets.load_iris()

ScatterStats(x=df["sepal_length"], y=df["sepal_width"]).plot()
```
