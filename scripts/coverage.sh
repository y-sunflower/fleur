uv run coverage run --source=fleur -m pytest
uv run coverage xml
uv run genbadge coverage -i coverage.xml
rm coverage.xml
