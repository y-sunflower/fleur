`fleur` aims to provide some differences, but its key feature is the ability to create impressive and easy-to-reproduce plots with very few lines of code. It is well-suited for both exploratory data analysis and more goal-oriented analysis. More generally, `fleur` tries to:

- being **super easy to use**
    * automatically detect which test to use
    * make nice plots by default
    * minimalist API
- letting you **control over everything**: both from the statistics and dataviz point of view
- being more **lightweight**: it only relies on
    * [`matplotlib`](https://matplotlib.org/){target="\_blank"}: for visualization
    * [`scipy`](https://scipy.org/){target="\_blank"}: for statistics
    * [`narwhals`](https://narwhals-dev.github.io/narwhals/){target="\_blank"}: for data handling (`fleur` accepts all inputs that `narhwals` support: `pandas`, `polars`, `pyarrow`, `cudf`, `modin`).
- provide an **extensive documentation** with many examples.

## Inspirations

`fleur` is highly inspired by the following projects:

- [`ggstatsplot`](https://indrajeetpatil.github.io/ggstatsplot/){target="\_blank"}: an R package that extends `ggplot2` to add statistical details to plots.
- [`seaborn`](https://seaborn.pydata.org/){target="\_blank"}: the famous high-level interface of matplotlib for statistical data visualization.
- [`statannotations`](https://github.com/trevismd/statannotations){target="\_blank"}: a `seaborn` extension that adds statistical annotations.
