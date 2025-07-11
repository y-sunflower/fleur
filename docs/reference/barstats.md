# Bar stats

::: fleur.barstats.BarStats

<br>

## Examples

- Minimalist example

```python
# mkdocs: render
from fleur import BarStats
from fleur import data

df = data.load_mtcars("pandas")

BarStats(x="cyl", y="vs", data=df).plot()
```

<br>
