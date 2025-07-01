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


@pytest.mark.parametrize("approach", ["parametric", "nonparametric"])
def test_three_categories(sample_data, approach):
    bs = BetweenStats(
        sample_data["x"],
        sample_data["y"],
        paired=False,
        approach=approach,
    )

    if approach == "parametric":
        assert bs.name == "One-way ANOVA"
    elif approach == "nonparametric":
        assert bs.name == "Kruskal-Wallis H-test"


@pytest.mark.parametrize("approach", ["parametric", "nonparametric"])
@pytest.mark.parametrize("paired", [True, False])
def test_two_categories(sample_data, paired, approach):
    sample_data = sample_data[sample_data["x"] != "setosa"]

    bs = BetweenStats(
        sample_data["x"],
        sample_data["y"],
        paired=paired,
        approach=approach,
    )

    assert hasattr(bs, "dof")

    if paired:
        if approach == "parametric":
            assert bs.name == "Paired t-test"
        elif approach == "nonparametric":
            assert bs.name == "Wilcoxon signed-rank test"
    else:
        if approach == "parametric":
            assert bs.name == "T-test"
        elif approach == "nonparametric":
            assert bs.name == "Mann-Whitney U rank test"


def test_not_enough_categories(sample_data):
    sample_data = sample_data[sample_data["x"] == "setosa"]

    with pytest.raises(
        ValueError,
        match="You must have at least 2 distinct categories in your category column",
    ):
        BetweenStats(sample_data["x"], sample_data["y"])


@pytest.mark.parametrize("approach", ["robust", "bayes"])
def test_raise_notimplemented_error(sample_data, approach):
    with pytest.raises(
        NotImplementedError,
        match="Repeated measures ANOVA has not been implemented yet.",
    ):
        BetweenStats(sample_data["x"], sample_data["y"], paired=True)

    with pytest.raises(
        NotImplementedError,
        match='Only `approach="parametric"` and `approach="nonparametric"` are implemented.',
    ):
        BetweenStats(sample_data["x"], sample_data["y"], approach=approach)


@pytest.mark.parametrize(
    "approach, paired, expected_exception, match",
    [
        (
            "bayes",
            False,
            NotImplementedError,
            (
                'Only `approach="parametric"`, `approach="nonparametric"` '
                'and `approach="robust"` have been implemented for '
                "independant samples."
            ),
        ),
        (
            "robust",
            True,
            NotImplementedError,
            (
                'Only `approach="parametric"` and `approach="nonparametric"` '
                "have been implemented for paired samples."
            ),
        ),
        (
            "bayes",
            True,
            NotImplementedError,
            (
                'Only `approach="parametric"` and `approach="nonparametric"` '
                "have been implemented for paired samples."
            ),
        ),
    ],
)
def test_raise_notimplemented_error2(
    sample_data, approach, paired, expected_exception, match
):
    sample_data = sample_data[sample_data["x"] != "setosa"]

    with pytest.raises(expected_exception, match=match):
        BetweenStats(
            sample_data["x"], sample_data["y"], approach=approach, paired=paired
        )


@pytest.mark.parametrize(
    "approach, trim, warning_match",
    [
        (
            "robust",
            None,
            (
                "Setting `approach='robust'` without setting a value "
                "of `trim` above 0 is equivalent of using default "
                "`approach='parametric'`. "
                "Remove `approach='robust'` to hide this warning."
            ),
        ),
        (
            "robust",
            0,
            (
                "Setting `approach='robust'` without setting a value "
                "of `trim` above 0 is equivalent of using default "
                "`approach='parametric'`. "
                "Remove `approach='robust'` to hide this warning."
            ),
        ),
    ],
)
def test_warns_for_robust_approach_without_trim(
    sample_data, approach, trim, warning_match
):
    sample_data = sample_data[sample_data["x"] != "setosa"]

    if trim is None:

        def warn_call():
            BetweenStats(sample_data["x"], sample_data["y"], approach=approach)
    else:

        def warn_call():
            BetweenStats(
                sample_data["x"], sample_data["y"], trim=trim, approach=approach
            )

    with pytest.warns(Warning, match=warning_match):
        warn_call()


def test_warn_trim_without_robust(sample_data):
    sample_data = sample_data[sample_data["x"] != "setosa"]

    with pytest.warns(
        UserWarning,
        match='Using `trim` argument without expliciting `approach="robust"` is not recommended.',
    ):
        BetweenStats(
            sample_data["x"], sample_data["y"], approach="parametric", trim=0.2
        )

    with pytest.raises(
        TypeError,
        match=re.escape("mannwhitneyu() got an unexpected keyword argument 'trim'"),
    ):
        with pytest.warns(
            UserWarning,
            match='Using `trim` argument without expliciting `approach="robust"` is not recommended.',
        ):
            BetweenStats(
                sample_data["x"], sample_data["y"], approach="nonparametric", trim=0.2
            )


def test_error_invalid_orientation(sample_data):
    with pytest.raises(
        ValueError,
        match="`orientation` must be one of: 'vertical', 'horizontal'.",
    ):
        BetweenStats(sample_data["x"], sample_data["y"]).plot(orientation="invalid")


def test_error_invalid_approach(sample_data):
    with pytest.raises(
        ValueError,
        match=r"^`approach` must be one of",
    ):
        BetweenStats(sample_data["x"], sample_data["y"], approach="invalid")


def test_error_invalid_colors(sample_data):
    with pytest.raises(
        ValueError,
        match=r"^`colors` argument must have at least",
    ):
        BetweenStats(sample_data["x"], sample_data["y"]).plot(colors=["#fff"])
