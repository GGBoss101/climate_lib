# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as get_version
sys.path.insert(0, os.path.abspath('..'))

# Safely extract the raw string from Hatch's automatically generated file
try:
    with open(os.path.abspath("../src/climate_lib/_version.py"), "r") as f:
        # Reads the file line looking for __version__ = "1.2.4"
        exec(f.read()) # This defines the local variable: __version__
    release = __version__
except Exception:
    release = '0.0.0 - unknown' # Clean local fallback if file isn't compiled yet

version = ".".join(release.split(".")[:2])

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'climate_lib'
copyright = '2026, Dhairya Chopra'
author = 'Dhairya Chopra'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
]

html_show_sourcelink = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

napoleon_custom_sections = [('Returns', 'params_style')]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']