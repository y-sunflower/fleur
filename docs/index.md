# `fleur`: stats and plots holding hands

`fleur` provides a set of tools to combine data visualization with statistics.

???+ warning

    fleur is still in a very early stage and in beta version: expect regular breaking changes.

<br>

## Examples

=== "Group comparison"

    ```py
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import datasets

    df = datasets.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot()
    ```

=== "Correlation"

    ```py
    # mkdocs: render
    from fleur import ScatterStats
    from fleur import datasets

    df = datasets.load_iris()

    ScatterStats(df["sepal_length"], df["sepal_width"]).plot()
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
