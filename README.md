# inferplot

**inferplot is currently a work in progress.**

Inferplot is basically [ggstatsplot](https://indrajeetpatil.github.io/ggstatsplot/index.html), but for Python.

I'm in the process of building a PoC to find out which library to use, to think about what form the graphs should take and so on. Any feedback is welcome (especially about the name, I'm not a big fan of inferplot).

The goal behind inferplot is:

- provide a fast way to do data exploration
- highly customizable plots but with a sufficient/clean default style
- dataframe agnostic thanks to narwhals

<br><br>

## Miscellaneous

- requires python 3.11

<br><br>

## State

| Functions        | Description                         | Parametric | Non-parametric | Robust | Bayesian |
| :--------------- | :---------------------------------- | :--------- | :------------- | :----- | :------- |
| `betweenstats()` | Between group/condition comparisons | ✅         | ❌             | ❌     | ❌       |
| `scatterstats()` | Correlation between two variables   | ✅         | ❌             | ❌     | ❌       |
