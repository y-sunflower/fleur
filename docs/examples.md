A bunch of examples of things you can do with `fleur`:

<br>

## Group comparison

=== "Default"

    ```py
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import datasets

    df = datasets.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot()
    ```

=== "Colors"

    ```py hl_lines="8"
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import datasets

    df = datasets.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot(
       colors=["#005f73", "#ee9b00", "#9b2226"]
    )
    ```

=== "Orientation"

    ```py hl_lines="8"
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import datasets

    df = datasets.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot(
       orientation="horizontal"
    )
    ```

=== "Hide statistics"

    ```py hl_lines="8"
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import datasets

    df = datasets.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot(
      show_stats=False
    )
    ```

=== "Hide chart elements"

    ```py hl_lines="8 9"
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

- `ScatterStats()`

```python
# mkdocs: render
from fleur import ScatterStats
from fleur import datasets

df = datasets.load_iris()

ScatterStats(x=df["sepal_length"], y=df["sepal_width"]).plot()
```
