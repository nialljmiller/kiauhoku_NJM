# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add the package's root directory to sys.path
sys.path.insert(0, os.path.abspath('../../kiauhoku'))

# Import the version from the package
from kiauhoku import __version__

project = 'kiauhoku'
copyright = '2025, Zachary R. Claytor'
author = 'Zachary R. Claytor'
release = __version__
root_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.napoleon',
    'nbsphinx',]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
