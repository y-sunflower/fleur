# `fleur`: combining statistics with visualization

<img src="https://github.com/JosephBARBIERDARNAL/static/blob/main/python-libs/fleur/image.png?raw=true" alt="fleur logo" align="right" width="150px"/>

With `fleur` (_pronounced "flur"_), statistics and data visualization are done at the same time. It's meant as a modern tool for highly detailed statistical annotations in plots with high customization capabilities.

It only requires foundational libraries: `matplotlib`, `scipy` and `narwhals`. Learn more [about fleur](./about.md).

???+ warning

    fleur is still in a very early stage: expect regular breaking changes.

<br>

## Examples

Currently, `fleur` offers 3 features that you can benefit from:

- `BetweenStats`: Use this when you want to **compare numerical data across categories** (e.g., customer satisfaction between two product versions)
- `ScatterStats`: Use this to explore the **correlation between numerical variables** (e.g., the relationship between age and salary)
- `BarStats`: Use this to examine the **relationship between categorical variables** (e.g., the relationship between gender and smoking status)

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

[See more examples](./examples/quick-start.md)

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
