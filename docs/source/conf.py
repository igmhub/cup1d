import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'cup1d'
copyright = '2024, Andreu Font-Ribera, Chris Pedersen, Jonas Chaves-Montero'
author = 'Andreu Font-Ribera, Chris Pedersen, Jonas Chaves-Montero'
release = '2024.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

html_theme = 'furo'
html_static_path = ['_static']
