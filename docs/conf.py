# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

def purge_module_contents(app: Sphinx, *args: Any) -> None:
    """Finds generated rst files and completely deletes the Module contents section."""
    for filename in os.listdir(app.srcdir):
        if not filename.endswith(".rst"):
            continue
            
        path = os.path.join(app.srcdir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split the file right where the Module contents section begins
        if "Module contents" in content:
            clean_content = content.split("Module contents")[0].strip() + "\n"
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(clean_content)

def setup(app: Sphinx) -> None:
    # Runs right before Sphinx starts building your HTML pages
    app.connect("builder-inited", purge_module_contents)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'climate_lib'
copyright = '2026, Dhairya Chopra'
author = 'Dhairya Chopra'
release = '0.0.1'

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
