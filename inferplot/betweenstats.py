# import matplotlib.pyplot as plt
# import narwhals as nw
# from narwhals.typing import IntoDataFrame


# def betweenstats(
#     x: str,
#     y: str,
#     data: IntoDataFrame,
#     ax=None,
# ):
#     data = nw.from_native(data)

#     if ax is None:
#         ax = plt.gca()

#     return ax


# if __name__ == "__main__":
#     from inferplot import datasets

#     data = datasets.load_data("iris")
#     fig, ax = plt.subplots()
#     betweenstats(data=data, x="species", y="sepal_length", ax=ax)
#     plt.savefig("cache.png", dpi=300)
