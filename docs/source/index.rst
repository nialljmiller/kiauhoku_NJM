.. kiauhoku documentation master file, created by
   sphinx-quickstart on Fri Mar 28 15:31:24 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: kiauhoku docs

.. rst-class:: frontpage

Kīauhōkū
========

Python utilities for stellar model grid interpolation.

| ©2025, `Zachary R. Claytor <https://claytorastro.wixsite.com/home>`_  
| Space Telescope Science Institute 

.. image:: https://img.shields.io/badge/ascl-2011.027-blue.svg?colorB=262255
   :target: https://ascl.net/2011.027
.. image:: https://badge.fury.io/gh/zclaytor%2Fkiauhoku.svg
   :target: https://badge.fury.io/gh/zclaytor%2Fkiauhoku
.. image:: https://badge.fury.io/py/kiauhoku.svg
   :target: https://badge.fury.io/py/kiauhoku
.. image:: https://img.shields.io/badge/read-the_paper-blue
   :target: https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C/abstract


| If you find this package useful, please cite `Claytor et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C/abstract>`_ and the `ASCL entry <https://ascl.net/2011.027>`_.
| Download the model grids from `Zenodo <https://doi.org/10.5281/zenodo.4287717>`_. 

| Kīauhōkū  
| From Hawaiian:

1. vt. To sense the span of a star's existence (i.e., its age).  
2. n. The speed of a star (in this case, its rotational speed).  

This name was created in partnership with Dr. Larry Kimura and Bruce Torres Fischer, a student participant in `A Hua He Inoa <https://imiloahawaii.org/a-hua-he-inoa>`_, a program to bring Hawaiian naming practices to new astronomical discoveries. We are grateful for their collaboration.

Kīauhōkū is a suite of Python tools to interact with, manipulate, and interpolate between stellar evolutionary tracks in a model grid. It was designed to work with the model grid used in `Claytor et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C/abstract>`_, which was generated using YREC with the magnetic braking law of `van Saders et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...776...67V/abstract>`_, but other stellar evolution model grids are available.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   about
   Quickstart <quickstart.ipynb>
   installed_grids/index
   custom_grids
   eep
   extensions/index
   tutorials/index
   reference/index
   papers
   contribute
