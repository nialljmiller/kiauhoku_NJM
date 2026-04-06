import os
import numpy as np
import pandas as pd
from tqdm import tqdm

name = 'JT2017'
raw_grids_path = os.path.expanduser('~/kiauhoku/kiauhoku/models/nodiff_out4z')

# Column indices (0-based) from the 84-column raw YREC files
# Based on header:
# Model #, shells, AGE(Gyr), log(L/Lsun), log(R/Rsun), log(g), log(Teff),
# Mconv.core, Mconv.env, R,T,Rho,P,kappa env,
# Central: log(T), log(RHO), log(P), BETA, ETA, X, Y, Z,
# Luminosity: ppI, ppII, ppIII, CNO, triple-alpha, He-C, gravity, neutrinos
# ... (neutrino/diagnostic cols 30-61) ...
# Xsurf(62), Ysurf(63), Zsurf(64), Z/X surf(65),
# ... more cols ...
# Tau(cz)(81), Mass(Msun)(82)

USECOLS = [2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 19, 20, 21, 22, 23, 24, 26, 27, 62, 63, 64, 65, 81, 82]

COLNAMES = [
    'Age(Gyr)',
    'log(L/Lsun)',
    'log(R/Rsun)',
    'logg',
    'Log Teff(K)',
    'mass_conv_core',
    'mass_conv_envelope',
    'logT(cen)',
    'logrho(cen)',
    'logP(cen)',
    'Xcen',
    'Ycen',
    'Zcen',
    'ppI',
    'ppII',
    'ppIII',
    'He_tripalpha',
    'He_HeC',
    'Xsurf',
    'Ysurf',
    'Zsurf',
    'surface_ZovrX',
    'Tau(cz)(sec)',
    'Mass(Msun)',
]

eep_params = dict(
    age='Age(Gyr)',
    log_central_temp='logT(cen)',
    core_hydrogen_frac='Xcen',
    hydrogen_lum='H lum (Lsun)',
    lum='L/Lsun',
    logg='logg',
    log_teff='Log Teff(K)',
    core_helium_frac='Ycen',
    teff_scale=20,
    lum_scale=1,
    intervals=[200, 50, 100, 100, 150],
)


def my_RGBump2(track, eep_params, i0=None):
    lum = eep_params['lum']
    N = len(track)
    lum_tr = track.loc[i0:, lum]
    RGBump = _first_true_index(lum_tr > 2.5) + 1
    if RGBump == 0:
        return -1
    return RGBump - 1


def _first_true_index(bools):
    if not bools.any():
        return -1
    return bools.idxmax()


def my_HRD(track, eep_params):
    Tscale = eep_params['teff_scale']
    Lscale = eep_params['lum_scale']
    logTeff = track[eep_params['log_teff']]
    logLum = track[eep_params['lum']]
    N = len(track)
    dist = np.zeros(N)
    for i in range(1, N):
        temp_dist = (((logTeff.iloc[i] - logTeff.iloc[i-1]) * Tscale) ** 2
                     + ((logLum.iloc[i] - logLum.iloc[i-1]) * Lscale) ** 2)
        dist[i] = dist[i-1] + np.sqrt(temp_dist)
    return dist


eep_functions = {'rgbump': my_RGBump2}
metric_function = my_HRD


def parse_filename(filename):
    file_str = filename.replace('_grnodf.track', '').replace('_grnodf', '')

    mass = float(file_str[1:4]) / 100.

    met_i = file_str.find('fh') + 2
    met_str = file_str[met_i:met_i + 4]
    met = float(met_str[1:]) / 100
    if met != 0 and met_str[0] == 'm':
        met *= -1

    alpha_i = file_str.find('al') + 2
    alpha_str = file_str[alpha_i:alpha_i + 2]
    alpha = float(alpha_str) / 10

    he_i = file_str.find('y') + 1
    he_str = file_str[he_i:he_i + 3]
    he = float(he_str) / 1000
    if he == 0.273:
        he = 0.272683

    ml_str = file_str[file_str.find('a') + 1:file_str.find('a') + 3]
    ml = float(ml_str) / 10.
    if ml == 1.7:
        ml = 1.724485
    if ml == 1.2:
        ml = 1.224485
    if ml == 2.2:
        ml = 2.224485

    return mass, met, alpha, he, ml


def from_yrec(path):
    fname = os.path.basename(path)
    initial_mass, initial_met, initial_alpha, initial_he, mixing_length = parse_filename(fname)

    data = np.loadtxt(path, comments='#', usecols=USECOLS)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    df = pd.DataFrame(data, columns=COLNAMES)

    # Convert log quantities to linear
    df['L/Lsun'] = 10 ** df['log(L/Lsun)']
    df['R/Rsun'] = 10 ** df['log(R/Rsun)']

    # Compute composite luminosities
    df['H lum (Lsun)'] = df['ppI'] + df['ppII'] + df['ppIII']
    df['He lum (Lsun)'] = df['He_tripalpha'] + df['He_HeC']

    # Drop intermediate cols we don't want in the final grid
    df = df.drop(columns=['log(L/Lsun)', 'log(R/Rsun)', 'ppI', 'ppII', 'ppIII', 'He_tripalpha', 'He_HeC'])

    n = len(df)
    s = np.arange(n)
    m  = np.ones(n) * initial_mass
    z  = np.ones(n) * initial_met
    al = np.ones(n) * initial_alpha
    he = np.ones(n) * initial_he
    ml = np.ones(n) * mixing_length

    multi_index = pd.MultiIndex.from_arrays(
        [m, z, al, he, ml, s],
        names=['initial_mass', 'initial_met', 'alpha_fe', 'initial_he', 'mixing_length', 'step']
    )
    df.index = multi_index

    return df


def setup(raw_grids_path=raw_grids_path, progress=True):
    filelist = sorted([
        f for f in os.listdir(raw_grids_path)
        if f.endswith('.track') and 'Empty' not in f
    ])

    df_list = []
    file_iter = tqdm(filelist) if progress else filelist
    for fname in file_iter:
        fpath = os.path.join(raw_grids_path, fname)
        try:
            df_list.append(from_yrec(fpath))
        except Exception as e:
            print(f'Skipping {fname}: {e}')

    dfs = pd.concat(df_list).sort_index()
    return dfs
