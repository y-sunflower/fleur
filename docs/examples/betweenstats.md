The `BetweenStats()` class is meant for comparing numerical values across multiple groups. You want to use it in cases such as:

- Blood glucose levels before and after treatment
- Customer satisfaction ratings between two product versions
- Weight changes across three fitness plans
- Air quality index in urban vs. rural areas

It supports tests for 2 groups or more, paired groups or not, equal variance or not, parametric or not, and a large set of styling options.

## Plot style

- Default

```py
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot()
```

- Change colors

```py hl_lines="8"
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
    colors=["#005f73", "#ee9b00", "#9b2226"]
)
```

- Change orientation

```py hl_lines="8"
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
    orientation="horizontal"
)
```

- Remove stats from plot

```py hl_lines="8"
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
    show_stats=False
)
```

- Remove means from plot

```py hl_lines="8"
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
    show_means=False
)
```

- Hide specific elements

```py hl_lines="8 9 10"
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
    box=False,
    scatter=False,
    violin=True, # default
)
```

- Advanced example

```python
# mkdocs: render
from fleur import BetweenStats, data
import polars as pl
import matplotlib.pyplot as plt

df = data.load_titanic("polars")
df = (
    df.select(pl.col("Age"), pl.col("Survived"))
    .drop_nulls()
    .with_columns(pl.col("Survived").cast(pl.String).cast(pl.Categorical))
    .with_columns(
        Survived=pl.when(pl.col("Survived") == "1")
        .then(pl.lit("Survived"))
        .otherwise(pl.lit("Died"))
    )
)

fig, ax = plt.subplots()
BetweenStats(df["Survived"], df["Age"], approach="nonparametric").plot(
    ax=ax,
    orientation="horizontal",
    scatter_kws={"alpha": 0.3, "s": 20},
    jitter_amount=0.3,
    colors=["#005f73", "#ee9b00"],
)
```

## Statistics

- Dependent (paired) samples

```py
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()
df = df[df["species"] != "setosa"] # keep only 2 groups

BetweenStats(df["sepal_length"], df["species"], paired=True).plot()
```

- Non-parametric test

```py
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()
df = df[df["species"] != "setosa"] # keep only 2 groups

BetweenStats(df["sepal_length"], df["species"], approach="nonparametric").plot()
```

- Non-parametric test + paired samples

```py hl_lines="11 12"
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()
df = df[df["species"] != "setosa"] # keep only 2 groups

BetweenStats(
    df["sepal_length"],
    df["species"],
    approach="nonparametric",
    paired=True,
).plot()
```

- Robust

```py hl_lines="11 12"
# mkdocs: render
from fleur import BetweenStats
from fleur import data

df = data.load_iris()
df = df[df["species"] != "setosa"] # keep only 2 groups

BetweenStats(
    df["sepal_length"],
    df["species"],
    approach="robust",
    trim=0.1,
).plot()
```

<br><br>

Learn more in the [`BetweenStats()` reference page](../../reference/betweenstats)
