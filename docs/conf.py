import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import vhr_composite

# Project information

project = 'vhr-composite'
copyright = '2023, Caleb S. Spradlin'
author = '2023, Caleb S. Spradlin'

# General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'jupyter_sphinx.execute',
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_click.ext",
    "sphinx.ext.githubpages",
    "nbsphinx",
]

intersphinx_mapping = {
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Options for HTML output

master_doc = "index"

version = release = vhr_composite.__version__

pygments_style = "sphinx"

todo_include_todos = False

html_theme = 'sphinx_rtd_theme'

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

myst_url_schemes = ("http", "https", "mailto")
