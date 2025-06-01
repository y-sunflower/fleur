# import pytest
# import matplotlib.pyplot as plt

# from inferplot import BetweenStats
# import inferplot.datasets as datasets


# @pytest.fixture
# def sample_data_pandas():
#     df = datasets.load_iris()
#     df = df.rename(columns={"species": "x", "sepal_length": "y"})
#     return df


# def test_summary_error():
#     with pytest.raises(RuntimeError):
#         BetweenStats.summary()


# def test_default(sample_data_pandas):
#     bs = BetweenStats.fit("x", "y", sample_data_pandas)
#     assert isinstance(bs.ax, plt.Axes), "Expected a matplotlib Axes object"
#     plt.close(bs.ax.figure)


# def test_custom_ax(sample_data_pandas):
#     fig, ax = plt.subplots()
#     bs = BetweenStats.fit("x", "y", sample_data_pandas, ax=ax)
#     assert bs.ax == ax, "Expected the returned Axes to be the same as the input Axes"
#     plt.close(fig)


# def test_invalid_columns(sample_data_pandas):
#     with pytest.raises(KeyError):
#         BetweenStats.fit("invalid_x", "y", sample_data_pandas)
#     with pytest.raises(KeyError):
#         BetweenStats.fit("x", "invalid_y", sample_data_pandas)


# def test_expected_attributes(sample_data_pandas):
#     fig, ax = plt.subplots()
#     bs = BetweenStats.fit("x", "y", sample_data_pandas, ax=ax)

#     assert hasattr(bs, "ax")
#     assert hasattr(bs, "statistic")
#     assert hasattr(bs, "pvalue")
#     assert hasattr(bs, "main_stat")
#     assert hasattr(bs, "expression")
#     assert hasattr(bs, "n_cat")


# if __name__ == "__main__":
#     pytest.main()
