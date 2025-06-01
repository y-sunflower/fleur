# Between stats

::: fleur.betweenstats.BetweenStats

<br>

## Examples

- Minimalist example

```python
# mkdocs: render
from fleur import BetweenStats
from fleur import datasets

df = datasets.load_iris()

BetweenStats(df["sepal_length"], df["species"]).plot()
```

<br>

## Statistical details

### Overview

The `BetweenStats` class is designed for **between-group statistical analysis**. It identifies whether there's a statistically significant difference between the means of numeric data across categorical groups. Internally, the class decides the appropriate test based on the number of groups:

- For **two groups**, it uses the **Student’s t-test** (paired or unpaired).
- For **three or more groups**, it uses a **one-way ANOVA**.

After computing the test, it returns relevant statistics like the test statistic, degrees of freedom, p-value, and a formatted LaTeX-style string for annotation in plots.

### Statistical Tests

#### 1. **Two Groups: t-test**

For two groups, we assess whether the difference in means is statistically significant using the t-distribution.

- **Unpaired (Independent) t-test**:

Let the two groups be $X_1, X_2, \ldots, X_{n_1}$ and $Y_1, Y_2, \ldots, Y_{n_2}$.

The test statistic is:

$$
t = \frac{\bar{X} - \bar{Y}}{\sqrt{\frac{s_X^2}{n_1} + \frac{s_Y^2}{n_2}}}
$$

Degrees of freedom (Welch–Satterthwaite equation):

$$
\nu = \frac{
  \left( \frac{s_X^2}{n_1} + \frac{s_Y^2}{n_2} \right)^2
}{
  \frac{ \left( \frac{s_X^2}{n_1} \right)^2 }{n_1 - 1} +
  \frac{ \left( \frac{s_Y^2}{n_2} \right)^2 }{n_2 - 1}
}
$$

- **Paired t-test**:

When the same subjects are measured twice (or matched pairs), the test is performed on the differences:

Let $D_i = X_i - Y_i$, then:

$$
t = \frac{\bar{D}}{s_D / \sqrt{n}}
$$

where $\bar{D}$ is the mean of differences and $s_D$ is the standard deviation of the differences.

Degrees of freedom:

$$
\nu = n - 1
$$

#### 2. **Three or More Groups: One-Way ANOVA**

When comparing more than two groups, we use **Analysis of Variance (ANOVA)** to test if at least one group mean differs from the others.

Let there be $k$ groups and $N$ total observations.

- **Between-group degrees of freedom**:

  $$
  df_{\text{between}} = k - 1
  $$

- **Within-group degrees of freedom**:

  $$
  df_{\text{within}} = N - k
  $$

- **F-statistic**:

$$
F = \frac{MS_{\text{between}}}{MS_{\text{within}}}
= \frac{ \frac{SS_{\text{between}}}{df_{\text{between}}} }{ \frac{SS_{\text{within}}}{df_{\text{within}}} }
$$

Where:

- $SS_{\text{between}}$ is the sum of squares between groups
- $SS_{\text{within}}$ is the sum of squares within groups
- $MS$ = Mean Square

If the null hypothesis (that all group means are equal) is true, the F-statistic follows the $F(df_{\text{between}}, df_{\text{within}})$ distribution.

### Reporting Results

The formatted statistical expression rendered on the plot is:

$$
\text{<Test Name>},\quad \text{<test statistic>},\quad p = \text{<p-value>},\quad n_{obs} = \text{<total n>}
$$

Examples:

- For t-test:

  $$
  t_{{Student}}(42) = 2.14,\quad p = 0.0391,\quad n_{{obs}} = 50
  $$

- For ANOVA:

  $$
  F(2, 57) = 5.82,\quad p = 0.0048,\quad n_{{obs}} = 60
  $$
