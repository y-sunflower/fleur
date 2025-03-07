from setuptools import setup

setup(
    name="inferplot",
    version="0.1.0",
    packages=["inferplot"],
    description="Add your description here",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Joseph Barbier",
    author_email="joseph.barbierdarnal@gmail.com",
    url="https://github.com/JosephBARBIERDARNAL/inferplot",
    install_requires=["matplotlib"],
    include_package_data=True,
)
