

<!-- Automatically generated, uses README.qmd to modify README.md -->

![Coverage](coverage-badge.svg)

# `fleur`: stats and plots holding hands

`fleur` provides a set of tools to combine data visualization with
statistics.

> fleur is still in a very early stage and in beta version: expect
> regular breaking changes.

[Documentation website](https://y-sunflower.github.io/fleur/)

<br>

## Quick start

### Group comparison

``` python
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

fig = BetweenStats(df["sepal_length"], df["species"]).plot()
```

![](README_files/figure-commonmark/cell-3-output-1.png)

### Correlation

``` python
from fleur import ScatterStats
from fleur import datasets

df = datasets.load_iris()

fig = ScatterStats(df["sepal_length"], df["sepal_width"]).plot()
```

![](README_files/figure-commonmark/cell-4-output-1.png)

<br><br>

## Installation

``` bash
pip install fleur
```

<br><br>
