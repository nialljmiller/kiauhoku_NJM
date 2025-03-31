About
=====

Installation
------------

``kiauhoku`` is installable using ``pip``:

.. code-block:: bash
   
   pip install kiauhoku

To get access to the latest features, you can install the development version from GitHub:

.. code-block:: bash

   git clone https://github.com/zclaytor/kiauhoku.git
   cd kiauhoku
   git switch dev
   pip install .

Citing
--------

If you find this package useful, please cite both `Claytor et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C/abstract>`_ and the `ASCL repository <https://ascl.net/2011.027>`_.

.. code-block:: bibtex

   @ARTICLE{Claytor2020,
         author = {{Claytor}, Zachary R. and {van Saders}, Jennifer L. and {Santos}, {\^A}ngela R.~G. and {Garc{\'\i}a}, Rafael A. and {Mathur}, Savita and {Tayar}, Jamie and {Pinsonneault}, Marc H. and {Shetrone}, Matthew},
         title = "{Chemical Evolution in the Milky Way: Rotation-based Ages for APOGEE-Kepler Cool Dwarf Stars}",
         journal = {\apj},
      keywords = {Stellar rotation, Stellar ages, Stellar evolution, Galaxy chemical evolution, 1629, 1581, 1599, 580, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Astrophysics of Galaxies},
            year = 2020,
         month = jan,
         volume = {888},
         number = {1},
            eid = {43},
         pages = {43},
            doi = {10.3847/1538-4357/ab5c24},
   archivePrefix = {arXiv},
         eprint = {1911.04518},
   primaryClass = {astro-ph.SR},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C},
         adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }

   @software{Claytor2020,
         author = {{Claytor}, Zachary R. and {van Saders}, Jennifer L. and {Santos}, {\^A}ngela R.~G. and {Garc{\'\i}a}, Rafael A. and {Mathur}, Savita and {Tayar}, Jamie and {Pinsonneault}, Marc H. and {Shetrone}, Matthew},
         title = "{kiauhoku: Stellar model grid interpolation}",
   howpublished = {Astrophysics Source Code Library, record ascl:2011.027},
            year = 2020,
         month = nov,
            eid = {ascl:2011.027},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2020ascl.soft11027C},
         adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }
