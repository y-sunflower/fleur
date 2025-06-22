# Between stats

# State

| Type           | No. of groups | Test           | Effect                        | Function used                   | Implemented |
| -------------- | ------------- | -------------- | ----------------------------- | ------------------------------- | ----------- |
| Parametric     | 2             | Student/Welch  | Cohen's d/Hedge's g           | Test:`scipy.stats.ttest_ind`    | ❌          |
| Non-parametric | 2             | Mann-Whitney U | r (rank-biserial correlation) | Test:`scipy.stats.mannwhitneyu` | ❌          |
| Robust         | 2             | Yuen           | Algina-Keselman-Penfield      | Test:`scipy.stats.ttest_ind`    | ❌          |

# Reference

::: fleur.betweenstats.BetweenStats

<br>

# Examples

- Minimalist example

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot()
```

<br>

- Change colors

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
  colors=["#005f73", "#ee9b00", "#9b2226"]
)
```

<br>

- Change orientation

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(
  orientation="horizontal"
)
```

<br>

- Remove elements

```python
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

- Hide statistics

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot(show_stats=False)
```

<br>

- Print summary statistics

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).summary()
```

```raw
Between stats comparison

Test: One-way ANOVA with 3 groups
F(2, 147) = 119.26, p = 0.0000, n_obs = 150
```

<br>
<br>

# Statistical details

When trying to compare groups, you should first answer the following questions:

- **Number of groups**: the two cases are when there are 2 groups and when there 3 or more groups.
- **Independence of sample**: are the group we're comparing the same person?
  - Paired groups: comparing the same people before and after giving them a drug
  - Independent groups: comparing a placebo and a treatment group
- **Data distribution**:
  - Normal distribution: we use parametric tests (rely on a statistical law)
    - **Equality of variance**: in parametric tests, we need to know if the variance in each group is the same or not
  - Not normal distribution: we use non-parametric tests (don't assume any statistical law)
- **Sample size**: a too small sample size (n < 30) can be an issue because we lack statistical power

## Comparing 2 groups

### Independent samples

There are 2 cases here: whether we assume the data distribution is normal or not. Many time, not assuming normality is more realistic, but it also reduces the power of the test (the probability of detecting a given effect if that effect actually exists).

=== "Parametric"

    Here we assume the data distribution is normal.

    - Equal variance: if the groups have equal variances: **independent t-test**.
    - Unequal variance: if the groups have unequal variances: **Welch's t-test**.

=== "Non-parametric"

    Here we don't assume anything about the distribution and we need to use the **Mann-Whitney U test**.

    Note that the **Mann-Whitney U test** compares distributions and not means. But this makes sense since not assuming normality (e.g having skewed distributions, for instance) implies that comparing means is not the best way to compare groups, which is what we want to do at the end.

### Dependent (paired) samples

=== "Parametric"

    Here we assume the data distribution is normal and we need to use a **paired t-test**.

=== "Non-parametric"

    Here we don't assume anything about the distribution and we need to use the **Wilcoxon signed-rank test**.

## Comparing 3 or more groups

### Independent samples

Again, there are parametric and non-parametric approaches depending on the assumption of normality. When normality is assumed, these tests compare group means; otherwise, they compare distributions more generally.

=== "Parametric"

    - Equal variance: if the groups have equal variances and normal distributions, use **one-way ANOVA**.
    - Unequal variance: if the groups have unequal variances, use **Welch’s ANOVA**.

=== "Non-parametric"

    Use the **Kruskal-Wallis test**, which does not assume normality and compares the overall distributions across groups.

### Dependent (repeated measures) samples

=== "Parametric"

    Assuming normality, use **repeated measures ANOVA** to compare means across related groups.

=== "Non-parametric"

    If normality is not assumed, use the **Friedman test**, which compares distributions across related groups without assuming normality.
