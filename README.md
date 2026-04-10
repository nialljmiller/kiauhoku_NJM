# [Kīauhōkū][kiauhoku github]

[![ascl:2011.027](https://img.shields.io/badge/ascl-2011.027-blue.svg?colorB=262255)](https://ascl.net/2011.027)
[![GitHub version](https://badge.fury.io/gh/zclaytor%2Fkiauhoku.svg)](https://badge.fury.io/gh/zclaytor%2Fkiauhoku)
[![PyPI version](https://badge.fury.io/py/kiauhoku.svg)](https://badge.fury.io/py/kiauhoku)
[![Paper](https://img.shields.io/badge/read-the_paper-blue)](https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C/abstract)
[![Docs](https://readthedocs.org/projects/kiauhoku/badge/?version=latest)](https://kiauhoku.readthedocs.io)
![Tests](https://github.com/zclaytor/kiauhoku/actions/workflows/run_tests.yml/badge.svg)

Python utilities for stellar model grid interpolation.

If you find this package useful, please cite [Claytor et al. (2020)][gyro paper].

Download the model grids from [Zenodo][zenodo].

(C) [Zachary R. Claytor][zclaytor]  
Space Telescope Science Institute  
2025 March 26

Kīauhōkū  
From Hawaiian:

1. vt. To sense the span of a star's existence (i.e., its age).  
2. n. The speed of a star (in this case, its rotational speed).  

This name was created in partnership with Dr. Larry Kimura and Bruce Torres Fischer, a student participant in *A Hua He Inoa*, a program to bring Hawaiian naming practices to new astronomical discoveries. We are grateful for their collaboration.

Kīauhōkū is a suite of Python tools to interact with, manipulate, and interpolate between stellar evolutionary tracks in a model grid. It was designed to work with the model grid used in [Claytor et al. (2020)][gyro paper], which was generated using YREC with the magnetic braking law of [van Saders et al. (2013)][van Saders], but other stellar evolution model grids are available.

## Installation

Kīauhōkū requires the use of Python 3 and uses the following Python packages:

- numpy
- scipy  
- pandas  
- matplotlib
- requests
- pyarrow (or some package that supports parquet files)
- numba
- [emcee][emcee]

Personally, I create a conda environment for this. In this example I'll call it "stars".

```bash
conda create -n stars numpy scipy pandas matplotlib requests pyarrow numba emcee
conda activate stars
pip install git+https://github.com/zclaytor/kiauhoku
```

Kīauhōkū is also on PyPI! It requires Python 3, but you can do this:

```bash
pip install kiauhoku
```

## Quickstart Guide

As of v.2.0, you no longer need to manually download and install model grids; `kiauhoku` will automatically download any missing grid from Zenodo when you try to load it!

```python
import kiauhoku as kh
grid = kh.load_interpolator('fastlaunch')
```

After this, the `fastlaunch` grid will be installed in `~/.kiauhoku/grids`. You shouldn't have to download it again after this. Note that download times will vary depending on the size of the model grid.

## How it works

We start with output evolution tracks from your favorite stellar modeling software. For `rotevol` output, these are the \*.out files. Each \*.out file has, for one specific initial metallicity and alpha-abundance, a series of evolution tracks for a range of initial masses. The "fastlaunch" grid for `kiauhoku` has eight \*.out files, corresponding to  
[M/H] ~ [-1.0, -0.5, 0.0, 0.5] and  
[alpha/M] ~ [0.0, 0.4].  
Each file contains 171 evolution tracks for 0.30 <= M/Msun <= 2.00 in steps of 0.01\*Msun.

1. First we load the tracks into a pandas MultiIndexed DataFrame and save to a parquet file.

2. Age is not an optimal dimension for comparing consecutive evolution tracks. For this reason we condense each evolution track in the time domain to a series of Equivalent Evolutionary Phases (EEPs) after the method of Dotter (2016). The EEP-based tracks are packaged into a MultiIndexed DataFrame and saved to parquet.

3. We finally load the EEP-based tracks into a `kiauhoku.stargrid.StarGridInterpolator` object. The `StarGridInterpolator` is based on the DataFrameInterpolator (`DFInterpolator`) from Tim Morton's [`isochrones`][isochrones] package. It performs linear interpolation between consecutive evolution tracks for an input mass, metallicity, alpha-abundance, and either age or EEP-index. We then pickle the interpolator so it can be accessed quickly and easily.

## Basic Usage

Once you have everything running, try doing this:  

```python
import kiauhoku as kh
grid = kh.load_interpolator('fastlaunch')
star = grid.get_star_eep((1, 0, 0, 330))
```

If it works, you should get something close to the sun. The argument to get_star_eep is a tuple containing the model grid indices. In this case, those are mass (in solar units), metallicity, alpha-abundance, and EEP index. See the documentation for more details.

Kīauhōkū comes with MCMC functionality through `emcee`. See the jupyter notebook `mcmc.ipynb` for an example.

## Installing Custom Model Grids

To install your own custom grid, you will want to create a setup script (see `custom_install.py` for an example). The only requirements are that your setup file contains (1) a function called `setup` that returns a pandas MultiIndexed DataFrame containing all your evolution tracks, (2) a variable `name` that is set to whatever you want your installed grid to be named, and (3) a variable `raw_grids_path` that sets the path to wherever your custom raw grid is downloaded.

The index for this DataFrame is what all the "get" functions will use to get and interpolate tracks and EEPs. Thus, if you want to access your grid using mass and metallicity, you'll want the DataFrame returned by `setup` to have mass and metallicity, as well as a column to represent the time/EEP step.

You can also use the setup file to define custom EEP functions (see `custom_install.my_RGBump`) for an example) and to tell `kiauhoku` which columns to use in its default EEP functions.

Once your setup file is ready, you can install your custom grid using

```python
import kiauhoku as kh
kh.install_grid('custom_install')
```

If you create a setup file for your favorite model grid and you'd like it to be public, create a pull request and I'll add you as a contributor!

## Papers that use Kīauhōkū

This is a list of papers that have used `kiauhoku` in their research. If you don't see your paper here, please let me know, or open a pull request!
<!-- 
This list is taken from https://ui.adsabs.harvard.edu/public-libraries/AKgEKEL5TDS8fCAVNvaqTA, exported with custom format:
%ZEncoding:html%zn. <A href="%u">%T</A>: %3.3G, %Y, %q, %V, %p.\n
-->

1. <A href="https://ui.adsabs.harvard.edu/abs/2022ApJ...930....7A">Rotation Distributions around the Kraft Break with TESS and Kepler: The Influences of Age, Metallicity, and Binarity</A>: Avallone, E. A., Tayar, J. N., van Saders, J. L., et al., 2022, ApJ, 930, 7.

2. <A href="https://ui.adsabs.harvard.edu/abs/2022ApJ...936..100B">Is [Y/Mg] a Reliable Age Diagnostic for FGK Stars?</A>: Berger, T. A., van Saders, J. L., Huber, D., et al., 2022, ApJ, 936, 100.

3. <A href="https://ui.adsabs.harvard.edu/abs/2024ApJ...970..166B">A New Asteroseismic Kepler Benchmark Constrains the Onset of Weakened Magnetic Braking in Mature Sun-like Stars</A>: Bhalotia, V., Huber, D., van Saders, J. L., et al., 2024, ApJ, 970, 166.

4. <A href="https://ui.adsabs.harvard.edu/abs/2023AJ....165...74B">Kepler-102: Masses and Compositions for a Super-Earth and Sub-Neptune Orbiting an Active Star</A>: Brinkman, C. L., Cadman, J., Weiss, L., et al., 2023, AJ, 165, 74.

5. <A href="https://ui.adsabs.harvard.edu/abs/2024RNAAS...8..201B">Identifying Uncertainties in Stellar Evolution Models Using the Open Cluster M67</A>: Byrom, S., Tayar, J., 2024, RNAAS, 8, 201.

6. <A href="https://ui.adsabs.harvard.edu/abs/2021ApJ...922..229C">TESS Asteroseismology of α Mensae: Benchmark Ages for a G7 Dwarf and Its M Dwarf Companion</A>: Chontos, A., Huber, D., Berger, T. A., et al., 2021, ApJ, 922, 229.

7. <A href="https://ui.adsabs.harvard.edu/abs/2024ApJ...962...47C">TESS Stellar Rotation up to 80 Days in the Southern Continuous Viewing Zone</A>: Claytor, Z. R., van Saders, J. L., Cao, L., et al., 2024, ApJ, 962, 47.

8. <A href="https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C">Chemical Evolution in the Milky Way: Rotation-based Ages for APOGEE-Kepler Cool Dwarf Stars</A>: Claytor, Z. R., van Saders, J. L., Santos, Â. R. G., et al., 2020, ApJ, 888, 43.

9. <A href="https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.5283G">The TIME Table: rotation and ages of cool exoplanet host stars</A>: Gaidos, E., Claytor, Z., Dungee, R., et al., 2023, MNRAS, 520, 5283.

10. <A href="https://ui.adsabs.harvard.edu/abs/2025arXiv250109095K">A Pair of Dynamically Interacting Sub-Neptunes Around TOI-6054</A>: Kroft, M. A., Beatty, T. G., Crossfield, I. J. M., et al., 2025, arXiv, arXiv:2501.09095.

11. <A href="https://ui.adsabs.harvard.edu/abs/2025AJ....169...47K">Two Earth-size Planets and an Earth-size Candidate Transiting the nearby Star HD 101581</A>: Kunimoto, M., Lin, Z., Millholland, S., et al., 2025, AJ, 169, 47.

12. <A href="https://ui.adsabs.harvard.edu/abs/2023ApJ...952..131M">Magnetic Activity Evolution of Solar-like Stars. I. S &lt;SUB&gt;ph&lt;/SUB&gt;-Age Relation Derived from Kepler Observations</A>: Mathur, S., Claytor, Z. R., Santos, Â. R. G., et al., 2023, ApJ, 952, 131.

13. <A href="https://ui.adsabs.harvard.edu/abs/2025arXiv250210109M">Magnetic activity evolution of solar-like stars: II. $S_{\rm ph}$-Ro evolution of Kepler main-sequence targets</A>: Mathur, S., Santos, A. R. G., Claytor, Z. R., et al., 2025, arXiv, arXiv:2502.10109.

14. <A href="https://ui.adsabs.harvard.edu/abs/2020ApJ...900..154M">The Evolution of Rotation and Magnetic Activity in 94 Aqr Aa from Asteroseismology with TESS</A>: Metcalfe, T. S., van Saders, J. L., Basu, S., et al., 2020, ApJ, 900, 154.

15. <A href="https://ui.adsabs.harvard.edu/abs/2021ApJ...921..122M">Magnetic and Rotational Evolution of ρ CrB from Asteroseismology with TESS</A>: Metcalfe, T. S., van Saders, J. L., Basu, S., et al., 2021, ApJ, 921, 122.

16. <A href="https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.4528M">Abundant sub-micron grains revealed in newly discovered extreme debris discs</A>: Moór, A., Ábrahám, P., Su, K. Y. L., et al., 2024, MNRAS, 528, 4528.

17. <A href="https://ui.adsabs.harvard.edu/abs/2025MNRAS.537...35R">NGTS-EB-7, an eccentric, long-period, low-mass eclipsing binary</A>: Rodel, T., Watson, C. A., Ulmer-Moll, S., et al., 2025, MNRAS, 537, 35.

18. <A href="https://ui.adsabs.harvard.edu/abs/2023A&amp;A...672A..56S">Temporal variation of the photometric magnetic activity for the Sun and Kepler solar-like stars</A>: Santos, A. R. G., Mathur, S., García, R. A., et al., 2023, A&amp;A, 672, A56.

19. <A href="https://ui.adsabs.harvard.edu/abs/2021AJ....162..215S">TESS-Keck Survey. V. Twin Sub-Neptunes Transiting the Nearby G Star HD 63935</A>: Scarsdale, N., Murphy, J. M. A., Batalha, N. M., et al., 2021, AJ, 162, 215.

20. <A href="https://ui.adsabs.harvard.edu/abs/2024RNAAS...8..166S">A Lack of Mass-gap Compact Object Binaries in APOGEE</A>: Schochet, M., Tayar, J., Andrews, J. J., 2024, RNAAS, 8, 166.

21. <A href="https://ui.adsabs.harvard.edu/abs/2022ApJ...927...31T">A Guide to Realistic Uncertainties on the Fundamental Properties of Solar-type Exoplanet Host Stars</A>: Tayar, J., Claytor, Z. R., Huber, D., et al., 2022, ApJ, 927, 31.

22. <A href="https://ui.adsabs.harvard.edu/abs/2022ApJ...930...78T">Potential Habitability as a Stellar Property: Effects of Model Uncertainties and Measurement Precision</A>: Tuchow, N. W., Wright, J. T., 2022, ApJ, 930, 78.

23. <A href="https://ui.adsabs.harvard.edu/abs/2022AJ....163..293T">The TESS-Keck Survey. XI. Mass Measurements for Four Transiting Sub-Neptunes Orbiting K Dwarf TOI-1246</A>: Turtelboom, E. V., Weiss, L. M., Dressing, C. D., et al., 2022, AJ, 163, 293.

24. <A href="https://ui.adsabs.harvard.edu/abs/2024MNRAS.535...90V">HD 28185 revisited: an outer planet, instead of a brown dwarf, on a Saturn-like orbit</A>: Venner, A., An, Q., Huang, C. X., et al., 2024, MNRAS, 535, 90.

25. <A href="https://ui.adsabs.harvard.edu/abs/2021AJ....161...56W">The TESS-Keck Survey. II. An Ultra-short-period Rocky Planet and Its Siblings Transiting the Galactic Thick-disk Star TOI-561</A>: Weiss, L. M., Dai, F., Huber, D., et al., 2021, AJ, 161, 56.



[kiauhoku github]: https://github.com/zclaytor/kiauhoku
[zclaytor]: https://claytorastro.wixsite.com/home
[gyro paper]: https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C/abstract
[van Saders]: https://ui.adsabs.harvard.edu/abs/2013ApJ...776...67V/abstract
[emcee]: https://emcee.readthedocs.io/en/latest/
[isochrones]: https://isochrones.readthedocs.io/en/latest/
[zenodo]: https://doi.org/10.5281/zenodo.4287717
