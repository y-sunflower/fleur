# Between stats

::: fleur.betweenstats.BetweenStats

<br>

## Examples

- Minimalist example

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot()
```

<br>

- Change colors

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
  colors=["#005f73", "#ee9b00", "#9b2226"]
)
```

<br>

- Change orientation

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
  orientation="horizontal"
)
```

<br>

- Remove elements

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
  box=False,
  scatter=False,
)
```

<br>

- Hide statistics

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(show_stats=False)
```

<br>

- Print summary statistics

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).summary()
```

```raw
Between stats comparison

Test: One-way ANOVA with 3 groups
F(2, 147) = 119.26, p = 0.0000, n_obs = 150
```

<br>
