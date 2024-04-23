
import os, sys

sys.path.insert(0, os.path.abspath(".."))

import automl_sphinx_theme  # Must come after the path injection above


options = {
    "copyright": copyright,
    "author": "Theresa Eimer",
    "version": "0.1",
    "versions": {
        f"v0.1 (stable)": "#",
    },
    "name": "Hypersweeper",
    "html_theme_options": {
        "github_url": "https://github.com/automl/automl_sphinx_theme",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    },
    # this is here to exclude the examples gallery since they are not documented
    "extensions": [
        "myst_parser",
        "sphinx.ext.autodoc",
        "sphinx.ext.viewcode",
        "sphinx.ext.napoleon",  # Enables to understand NumPy docstring
        # "numpydoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.autosectionlabel",
        "sphinx_autodoc_typehints",
        "sphinx.ext.doctest",
    ],
}

# Import conf.py from the automl theme
automl_sphinx_theme.set_options(globals(), options)