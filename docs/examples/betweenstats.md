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
    paired=True
).plot()
```

<br><br>

Learn more in the [`BetweenStats()` reference page](../../reference/betweenstats)
