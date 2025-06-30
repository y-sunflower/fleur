The easiest way to get started is to use [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### Install for development

- Fork the repository to your own GitHub account.

- Clone your forked repository to your local machine:

```bash
git clone https://github.com/YOURNAME/fleur.git
cd fleur
git remote add upstream https://github.com/y-sunflower/fleur.git
```

- Create a new branch:

```bash
git checkout -b my-feature
```

- Set up your Python environment:

```bash
uv sync --all-groups
uv run pre-commit install
uv pip install -e .
```

### Code!

You can now make changes to the package and start coding!

### Run the test

- Test that everything works correctly by running:

```bash
uv run pytest
```

### Preview documentation locally

```bash
uv run mkdocs serve
```

### Push changes

- Commit and push your changes:

```bash
git add -A
git commit -m "description of what you did"
git push
```

- Go back to your fork and click on the "Open a PR" popup

Congrats! Once your PR is merged, it will be part of `fleur`.

<br>
