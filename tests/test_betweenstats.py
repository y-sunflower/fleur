import pytest
import re
import matplotlib.pyplot as plt
import matplotlib as mpl

from fleur import BetweenStats
import fleur.datasets as datasets


@pytest.fixture
def sample_data():
    df = datasets.load_iris()
    df = df.rename(columns={"species": "x", "sepal_length": "y"})
    return df


def test_default(sample_data):
    fig = BetweenStats(sample_data["x"], sample_data["y"]).plot()
    assert isinstance(fig, mpl.figure.Figure), (
        f"Expected a matplotlib Figure, not: {type(fig)}"
    )
    plt.close(fig)


def test_custom_ax(sample_data):
    fig, ax = plt.subplots()
    bs = BetweenStats(sample_data["x"], sample_data["y"])
    bs.plot(ax=ax)
    assert bs.ax == ax, "Expected the returned Axes to be the same as the input Axes"
    plt.close(fig)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
@pytest.mark.parametrize("show_stats", [True, False])
def test_expected_attributes(sample_data, orientation, show_stats):
    fig, ax = plt.subplots()
    bs = BetweenStats(sample_data["x"], sample_data["y"])
    bs.plot(orientation=orientation, show_stats=show_stats, ax=ax)

    assert hasattr(bs, "ax")
    assert hasattr(bs, "statistic")
    assert hasattr(bs, "pvalue")
    assert hasattr(bs, "main_stat")
    assert hasattr(bs, "n_cat")
    assert hasattr(bs, "n_obs")
    assert hasattr(bs, "name")
    assert hasattr(bs, "dof_between")
    assert hasattr(bs, "dof_within")

    assert bs.name == "One-way ANOVA"

    plt.close(fig)


def test_summary_prints(capsys, sample_data):
    bs = BetweenStats(sample_data["x"], sample_data["y"])
    bs.summary()
    captured = capsys.readouterr()
    assert captured.out.startswith("Between stats comparison")


@pytest.mark.parametrize("method", ["parametric", "nonparametric"])
def test_three_categories(sample_data, method):
    bs = BetweenStats(
        sample_data["x"],
        sample_data["y"],
        paired=False,
        method=method,
    )

    if method == "parametric":
        assert bs.name == "One-way ANOVA"
    elif method == "nonparametric":
        assert bs.name == "Kruskal-Wallis H-test"


@pytest.mark.parametrize("method", ["parametric", "nonparametric"])
@pytest.mark.parametrize("paired", [True, False])
def test_two_categories(sample_data, paired, method):
    sample_data = sample_data[sample_data["x"] != "setosa"]

    bs = BetweenStats(
        sample_data["x"],
        sample_data["y"],
        paired=paired,
        method=method,
    )

    assert hasattr(bs, "dof")

    if paired:
        if method == "parametric":
            assert bs.name == "Paired t-test"
        elif method == "nonparametric":
            assert bs.name == "Wilcoxon signed-rank test"
    else:
        if method == "parametric":
            assert bs.name == "T-test"
        elif method == "nonparametric":
            assert bs.name == "Mann-Whitney U rank test"


def test_not_enough_categories(sample_data):
    sample_data = sample_data[sample_data["x"] == "setosa"]

    with pytest.raises(
        ValueError,
        match="You must have at least 2 distinct categories in your category column",
    ):
        BetweenStats(sample_data["x"], sample_data["y"])


@pytest.mark.parametrize("method", ["robust", "bayes"])
def test_raise_notimplemented_error(sample_data, method):
    with pytest.raises(
        NotImplementedError,
        match="Repeated measures ANOVA has not been implemented yet.",
    ):
        BetweenStats(sample_data["x"], sample_data["y"], paired=True)

    with pytest.raises(
        NotImplementedError,
        match='Only `method="parametric"` and `method="nonparametric"` are implemented.',
    ):
        BetweenStats(sample_data["x"], sample_data["y"], method=method)


@pytest.mark.parametrize(
    "method, paired, expected_exception, match",
    [
        (
            "bayes",
            False,
            NotImplementedError,
            (
                'Only `method="parametric"`, `method="nonparametric"` '
                'and `method="robust"` have been implemented for '
                "independant samples."
            ),
        ),
        (
            "robust",
            True,
            NotImplementedError,
            (
                'Only `method="parametric"` and `method="nonparametric"` '
                "have been implemented for paired samples."
            ),
        ),
        (
            "bayes",
            True,
            NotImplementedError,
            (
                'Only `method="parametric"` and `method="nonparametric"` '
                "have been implemented for paired samples."
            ),
        ),
    ],
)
def test_raise_notimplemented_error2(
    sample_data, method, paired, expected_exception, match
):
    sample_data = sample_data[sample_data["x"] != "setosa"]

    with pytest.raises(expected_exception, match=match):
        BetweenStats(sample_data["x"], sample_data["y"], method=method, paired=paired)


@pytest.mark.parametrize(
    "method, trim, warning_match",
    [
        (
            "robust",
            None,
            (
                "Setting `method='robust'` without setting a value "
                "of `trim` above 0 is equivalent of using default "
                "`method='parametric'`. "
                "Remove `method='robust'` to hide this warning."
            ),
        ),
        (
            "robust",
            0,
            (
                "Setting `method='robust'` without setting a value "
                "of `trim` above 0 is equivalent of using default "
                "`method='parametric'`. "
                "Remove `method='robust'` to hide this warning."
            ),
        ),
    ],
)
def test_warns_for_robust_method_without_trim(sample_data, method, trim, warning_match):
    sample_data = sample_data[sample_data["x"] != "setosa"]

    if trim is None:

        def warn_call():
            BetweenStats(sample_data["x"], sample_data["y"], method=method)
    else:

        def warn_call():
            BetweenStats(sample_data["x"], sample_data["y"], trim=trim, method=method)

    with pytest.warns(Warning, match=warning_match):
        warn_call()


def test_warn_trim_without_robust(sample_data):
    sample_data = sample_data[sample_data["x"] != "setosa"]

    with pytest.warns(
        UserWarning,
        match='Using `trim` argument without expliciting `method="robust"` is not recommended.',
    ):
        BetweenStats(sample_data["x"], sample_data["y"], method="parametric", trim=0.2)

    with pytest.raises(
        TypeError,
        match=re.escape("mannwhitneyu() got an unexpected keyword argument 'trim'"),
    ):
        with pytest.warns(
            UserWarning,
            match='Using `trim` argument without expliciting `method="robust"` is not recommended.',
        ):
            BetweenStats(
                sample_data["x"], sample_data["y"], method="nonparametric", trim=0.2
            )


def test_error_invalid_orientation(sample_data):
    with pytest.raises(
        ValueError,
        match="`orientation` must be one of: 'vertical', 'horizontal'.",
    ):
        BetweenStats(sample_data["x"], sample_data["y"]).plot(orientation="invalid")


def test_error_invalid_method(sample_data):
    with pytest.raises(
        ValueError,
        match=r"^`method` must be one of",
    ):
        BetweenStats(sample_data["x"], sample_data["y"], method="invalid")


def test_error_invalid_colors(sample_data):
    with pytest.raises(
        ValueError,
        match=r"^`colors` argument must have at least",
    ):
        BetweenStats(sample_data["x"], sample_data["y"]).plot(colors=["#fff"])
