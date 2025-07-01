# Scatter stats

::: fleur.scatterstats.ScatterStats

<br>

## Examples

- Minimalist example

```python
# mkdocs: render
from fleur import ScatterStats
from fleur import data

df = data.load_iris()

ScatterStats(x=df["sepal_length"], y=df["sepal_width"]).plot()
```

<br>
