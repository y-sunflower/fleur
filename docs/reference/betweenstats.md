# Between stats

::: fleur.betweenstats.BetweenStats

<br>

# Examples

=== "Default"

    ```py
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import data

    df = data.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot()
    ```

=== "Colors"

    ```py hl_lines="8"
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import data

    df = data.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot(
       colors=["#005f73", "#ee9b00", "#9b2226"]
    )
    ```

=== "Orientation"

    ```py hl_lines="8"
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import data

    df = data.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot(
       orientation="horizontal"
    )
    ```

=== "Hide statistics"

    ```py hl_lines="8"
    # mkdocs: render
    from fleur import BetweenStats
    from fleur import data

    df = data.load_iris()

    BetweenStats(df["sepal_length"], df["species"]).plot(
      show_stats=False
    )
    ```

=== "Hide chart elements"

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

<br>
<br>

# Statistical details

✅ means it's already implemented in `fleur`.

❌ means it's not implemented in `fleur` yet.

## Comparing 2 groups

### Independent samples

There are 2 cases here: whether we assume the data distribution is normal or not. Many time, not assuming normality is more realistic, but it also reduces the power of the test (the probability of detecting a given effect if that effect actually exists).

=== "Parametric ✅"

    Here we assume the data distribution is normal.

    - Equal variance: if the groups have equal variances: **independent t-test**.
    - Unequal variance: if the groups have unequal variances: **Welch's t-test**.

=== "Non-parametric ✅"

    Here we don't assume anything about the distribution and we need to use the **Mann-Whitney U test**.

    Note that the **Mann-Whitney U test** compares distributions and not means. But this makes sense since not assuming normality (e.g having skewed distributions, for instance) implies that comparing means is not the best way to compare groups, which is what we want to do at the end.

=== "Robust ✅"

    Here we don't assume anything about the distribution and we need to use the **"Yuen's t-test"**.

### Dependent (paired) samples

=== "Parametric ✅"

    Here we assume the data distribution is normal and we need to use a **paired t-test**.

=== "Non-parametric ✅"

    Here we don't assume anything about the distribution and we need to use the **Wilcoxon signed-rank test**.

=== "Robust ❌"

    Here we don't assume anything about the distribution and we need to use the **"Yuen's t-test"** for dependent samples.

## Comparing 3 or more groups

### Independent samples

Again, there are parametric and non-parametric approaches depending on the assumption of normality. When normality is assumed, these tests compare group means; otherwise, they compare distributions more generally.

=== "Parametric ✅"

    - Equal variance: if the groups have equal variances and normal distributions, use **one-way ANOVA**.
    - Unequal variance: if the groups have unequal variances, use **Welch’s ANOVA**.

=== "Non-parametric ✅"

    Use the **Kruskal-Wallis test**, which does not assume normality and compares the overall distributions across groups.

=== "Robust ❌"

    TODO

### Dependent (repeated measures) samples

=== "Parametric ❌"

    Assuming normality, use **repeated measures ANOVA** to compare means across related groups.

=== "Non-parametric ❌"

    If normality is not assumed, use the **Friedman test**, which compares distributions across related groups without assuming normality.

=== "Robust ❌"

    TODO
