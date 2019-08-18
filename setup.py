import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="feature-extract",
    version="0.0.1",
    author="Iñigo Sanchez Mendez",
    author_email="jisanchez003@ikasle.ehu.es",
    description="Package for feature extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spolex/pyrestfmri",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)
