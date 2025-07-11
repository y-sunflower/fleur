# `fleur`: stats and plots holding hands

`fleur` provides a set of tools to combine data visualization with statistics.

???+ warning

    fleur is still in a very early stage and in beta version: expect regular breaking changes.

<br>

## Examples

Currently, `fleur` has 3 things that you can benefit:

- `BetweenStats`: a class for group comparisons
- `ScatterStats`: a class for numerical correlation
- `BarStats`: a class for categorical data comparisons

=== "Group comparison"

    ```py
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import data

    df = data.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot()
    ```

=== "Correlation"

    ```py
    # mkdocs: render
    from fleur import ScatterStats
    from fleur import data

    df = data.load_iris()

    ScatterStats(df["sepal_length"], df["sepal_width"]).plot()
    ```

=== "Categorical data"

    ```py
    # mkdocs: render
    from fleur import BarStats
    from fleur import data

    df = data.load_mtcars()

    BarStats(df["cyl"], df["vs"]).plot()
    ```

## Installation

=== "stable"

    ```bash
    pip install fleur
    ```

=== "dev"

    ```bash
    pip install git+https://github.com/y-sunflower/fleur.git
    ```

<br><br>
