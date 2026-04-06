'''
fit_platinum_sample.py

Batch fit all stars in platinum_sample_flame.fits using the JT2017t11 kiauhoku grid.

Outputs
-------
results/fit_results.csv          — table of best-fit parameters for all stars
results/diagnostics/star_N.png  — per-star diagnostic plot
results/summary_plots.png        — summary plots across the full sample
'''

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import warnings
import kiauhoku as kh

warnings.filterwarnings('ignore')

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs('results/diagnostics', exist_ok=True)

# ── Load the grid ─────────────────────────────────────────────────────────────
print("Loading JT2017t11 grid...")
qstring = '201 <= eep'
evolve_met = False

jtgrid = kh.load_eep_grid("JT2017t11").query(qstring)
jtgrid['mass']          = jtgrid['Mass(Msun)']
jtgrid['teff']          = 10**jtgrid['Log Teff(K)']
jtgrid['lum']           = jtgrid['L/Lsun']
jtgrid['met']           = jtgrid.index.get_level_values('initial_met')
jtgrid['initial_he']    = jtgrid.index.get_level_values('initial_he')
jtgrid['mixing_length'] = jtgrid.index.get_level_values('mixing_length')
jtgrid['alpha_fe']      = jtgrid.index.get_level_values('alpha_fe')
jtgrid['age']           = jtgrid['Age(Gyr)']
jtgrid.set_name('jtgrid')
jtgrid = jtgrid.to_interpolator()
print("Grid loaded.\n")

# ── Grid observable ranges (for pre-filtering) ────────────────────────────────
# Stars clearly outside these ranges will not converge and may segfault.
# Determined from the grid index extents.
GRID_TEFF_MIN = 3500.0   # K  — conservative lower bound
GRID_TEFF_MAX = 12000.0  # K  — conservative upper bound
GRID_LOGG_MIN = -0.5
GRID_LOGG_MAX = 4.5
GRID_MH_MIN   = -2.6
GRID_MH_MAX   = 0.6

# ── Helium-metallicity relation ───────────────────────────────────────────────
# Standard linear enrichment law: Y = Y_p + (dY/dZ)*Z
# Y_p = 0.2485 (Serenelli & Basu 2010), dY/dZ = 1.4
# Z = Z_solar * 10^[M/H], Z_solar = 0.0134
Y_PRIMORDIAL = 0.2485
DYDZ         = 1.4
Z_SOLAR      = 0.0134

def compute_initial_he(mh):
    Z = Z_SOLAR * 10**mh
    Y = Y_PRIMORDIAL + DYDZ * Z
    return np.clip(Y, 0.24, 0.32)

# ── Fitting scale ─────────────────────────────────────────────────────────────
# Scale factors tell the optimizer how much each parameter matters relative
# to its typical uncertainty. Smaller scale = stronger constraint.
SCALE = {
    'teff':     100.0,   # K  — typical APOGEE spectroscopic uncertainty
    'logg':     0.10,    # dex
    'met':      0.10,    # dex [M/H]
    'alpha_fe': 0.10,    # dex [α/M]
    'lum':      0.10,    # dex log(L/L☉)
}

FIT_TOL = 1e-3

# ── Load the FITS catalogue ───────────────────────────────────────────────────
print("Reading platinum_sample_flame.fits...")
with fits.open('platinum_sample_flame.fits') as hdul:
    cat = hdul[1].data

n_stars = len(cat)
print(f"  {n_stars} stars to fit.\n")

# ── Containers for results ────────────────────────────────────────────────────
result_rows = []

# ── Per-star fitting loop ─────────────────────────────────────────────────────
for i, row in enumerate(cat):
    star_id   = row['sdss4_apogee_id'].strip()
    teff_obs  = float(row['teff_astra'])
    e_teff    = float(row['e_teff_astra'])
    logg_obs  = float(row['logg_astra'])
    e_logg    = float(row['e_logg_astra'])
    mh_obs    = float(row['mh_astra'])
    e_mh      = float(row['e_mh_astra'])
    alpha_obs = float(row['alpha_m_astra'])
    e_alpha   = float(row['e_alpha_m_astra'])
    lum_obs   = float(row['log_lum_lsun'])

    he_est = compute_initial_he(mh_obs)

    # Pre-check: skip stars clearly outside the grid
    out_of_grid = (
        teff_obs < GRID_TEFF_MIN or teff_obs > GRID_TEFF_MAX or
        logg_obs < GRID_LOGG_MIN or logg_obs > GRID_LOGG_MAX or
        mh_obs   < GRID_MH_MIN   or mh_obs   > GRID_MH_MAX
    )
    if out_of_grid:
        print(f"[{i+1:3d}/{n_stars}] {star_id}  SKIPPED — observables outside grid range "
              f"(Teff={teff_obs:.0f}, logg={logg_obs:.2f}, [M/H]={mh_obs:.2f})")
        res = {
            'star_id': star_id, 'ra': float(row['ra']), 'dec': float(row['dec']),
            'teff_obs': teff_obs, 'e_teff_obs': e_teff,
            'logg_obs': logg_obs, 'e_logg_obs': e_logg,
            'mh_obs': mh_obs, 'e_mh_obs': e_mh,
            'alpha_obs': alpha_obs, 'e_alpha_obs': e_alpha,
            'lum_obs': lum_obs, 'he_est': he_est,
            'fit_success': False, 'fit_loss': np.nan, 'fit_converged': False,
            'skip_reason': 'out_of_grid',
        }
        for param in ['initial_mass','initial_met','initial_he','alpha_fe',
                      'mixing_length','eep','mass','teff','lum','met','logg','age']:
            res[f'fit_{param}'] = np.nan
        for param in ['age','mass','mixing_length','initial_met']:
            res[f'nn_std_{param}'] = np.nan
        res['nn_dist_best'] = np.nan
        result_rows.append(res)
        continue

    print(f"[{i+1:3d}/{n_stars}] {star_id}  Teff={teff_obs:.0f}K  logg={logg_obs:.2f}  "
          f"[M/H]={mh_obs:.2f}  lum={lum_obs:.2f}")

    star_dict = {
        'teff':     teff_obs,
        'logg':     logg_obs,
        'met':      mh_obs,
        'alpha_fe': alpha_obs,
        'lum':      lum_obs,
        'initial_he': he_est,
    }
    # Scale initial_he tightly — it's computed from metallicity, not free
    scale = dict(SCALE)
    scale['initial_he'] = 0.005

    # ── Fit ──────────────────────────────────────────────────────────────────
    try:
        model, fit = jtgrid.gridsearch_fit(
            star_dict, scale=scale, tol=FIT_TOL, verbose=False
        )
    except Exception as exc:
        print(f"    ERROR: {exc}")
        model, fit = None, None

    # ── Nearest neighbours (for uncertainty estimate) ─────────────────────
    try:
        nn = jtgrid.nearest_match(star_dict, n=20, scale=scale)
    except Exception:
        nn = None

    # ── Pack results ─────────────────────────────────────────────────────────
    res = {
        'skip_reason':   'none',
        'star_id':       star_id,
        'ra':            float(row['ra']),
        'dec':           float(row['dec']),
        # Observations
        'teff_obs':      teff_obs,
        'e_teff_obs':    e_teff,
        'logg_obs':      logg_obs,
        'e_logg_obs':    e_logg,
        'mh_obs':        mh_obs,
        'e_mh_obs':      e_mh,
        'alpha_obs':     alpha_obs,
        'e_alpha_obs':   e_alpha,
        'lum_obs':       lum_obs,
        'he_est':        he_est,
        # Fit quality
        'fit_success':   fit.success if fit is not None else False,
        'fit_loss':      float(fit.fun) if fit is not None else np.nan,
        'fit_converged': getattr(fit, 'fun', np.nan) <= FIT_TOL if fit is not None else False,
    }

    if model is not None:
        for param in ['initial_mass', 'initial_met', 'initial_he', 'alpha_fe',
                      'mixing_length', 'eep', 'mass', 'teff', 'lum', 'met',
                      'logg', 'age']:
            res[f'fit_{param}'] = float(model[param]) if param in model.index else np.nan
    else:
        for param in ['initial_mass', 'initial_met', 'initial_he', 'alpha_fe',
                      'mixing_length', 'eep', 'mass', 'teff', 'lum', 'met',
                      'logg', 'age']:
            res[f'fit_{param}'] = np.nan

    # ── Uncertainty from nearest-neighbour spread ─────────────────────────
    if nn is not None:
        top5 = nn.head(5)
        for param in ['age', 'mass', 'mixing_length', 'initial_met']:
            if param in top5.columns:
                res[f'nn_std_{param}'] = float(top5[param].std())
            else:
                res[f'nn_std_{param}'] = np.nan
        res['nn_dist_best'] = float(nn.iloc[0]['distance'])
    else:
        for param in ['age', 'mass', 'mixing_length', 'initial_met']:
            res[f'nn_std_{param}'] = np.nan
        res['nn_dist_best'] = np.nan

    result_rows.append(res)

    # ── Diagnostic plot for this star ─────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"{star_id}   [{i+1}/{n_stars}]", fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_hrd   = fig.add_subplot(gs[0, 0])
    ax_kiel  = fig.add_subplot(gs[0, 1])
    ax_resid = fig.add_subplot(gs[0, 2])
    ax_nn_age = fig.add_subplot(gs[1, 0])
    ax_nn_mass = fig.add_subplot(gs[1, 1])
    ax_nn_ml   = fig.add_subplot(gs[1, 2])

    # --- Panel 1: HR diagram (lum vs teff) ---
    ax_hrd.set_title("H-R Diagram", fontsize=10)
    if nn is not None:
        ax_hrd.scatter(nn['teff'], nn['lum'], c=nn['distance'],
                       cmap='YlOrRd_r', s=30, zorder=2, label='Top 20 NN')
    ax_hrd.errorbar(teff_obs, lum_obs, xerr=e_teff, fmt='ko', ms=7,
                    zorder=5, label='Observed')
    if model is not None and not np.isnan(res['fit_teff']):
        ax_hrd.scatter(res['fit_teff'], res['fit_lum'], marker='*', s=200,
                       c='dodgerblue', zorder=6, label='Best fit')
    ax_hrd.set_xlabel('Teff (K)', fontsize=9)
    ax_hrd.set_ylabel('log(L/L☉)', fontsize=9)
    ax_hrd.invert_xaxis()
    ax_hrd.legend(fontsize=7)

    # --- Panel 2: Kiel diagram (logg vs teff) ---
    ax_kiel.set_title("Kiel Diagram", fontsize=10)
    if nn is not None:
        ax_kiel.scatter(nn['teff'], nn['logg'], c=nn['distance'],
                        cmap='YlOrRd_r', s=30, zorder=2)
    ax_kiel.errorbar(teff_obs, logg_obs, xerr=e_teff, yerr=e_logg,
                     fmt='ko', ms=7, zorder=5)
    if model is not None and not np.isnan(res['fit_teff']):
        ax_kiel.scatter(res['fit_teff'], res['fit_logg'], marker='*',
                        s=200, c='dodgerblue', zorder=6)
    ax_kiel.set_xlabel('Teff (K)', fontsize=9)
    ax_kiel.set_ylabel('log g', fontsize=9)
    ax_kiel.invert_xaxis()
    ax_kiel.invert_yaxis()

    # --- Panel 3: Observable residuals ---
    ax_resid.set_title("Fit Residuals (obs − fit)", fontsize=10)
    observables = ['teff', 'logg', 'lum', 'met', 'alpha_fe']
    obs_vals    = [teff_obs, logg_obs, lum_obs, mh_obs, alpha_obs]
    obs_errs    = [e_teff, e_logg, 0.05, e_mh, e_alpha]
    obs_labels  = ['Teff', 'logg', 'lum', '[M/H]', '[α/M]']
    fit_keys    = ['fit_teff', 'fit_logg', 'fit_lum', 'fit_met', 'fit_alpha_fe']
    resids, pull_colors = [], []
    for obs, err, fkey, scale_val, label in zip(
            obs_vals, obs_errs, fit_keys,
            [SCALE['teff'], SCALE['logg'], SCALE['lum'], SCALE['met'], SCALE['alpha_fe']],
            obs_labels):
        fval = res.get(fkey, np.nan)
        if not np.isnan(fval):
            resids.append((obs - fval) / scale_val)
            pull_colors.append('dodgerblue')
        else:
            resids.append(np.nan)
            pull_colors.append('grey')
    y_pos = range(len(obs_labels))
    for y, r, c in zip(y_pos, resids, pull_colors):
        if not np.isnan(r):
            ax_resid.barh(y, r, color=c, alpha=0.8)
    ax_resid.axvline(0, color='k', lw=0.8)
    ax_resid.axvline(-1, color='grey', lw=0.6, ls='--')
    ax_resid.axvline(1, color='grey', lw=0.6, ls='--')
    ax_resid.set_yticks(list(y_pos))
    ax_resid.set_yticklabels(obs_labels, fontsize=9)
    ax_resid.set_xlabel('(obs − fit) / scale', fontsize=9)
    ax_resid.set_title("Residuals (|<1| = good)", fontsize=10)

    # --- Panels 4-6: NN parameter distributions ---
    for ax, param, label, color in [
        (ax_nn_age,  'age',           'Age (Gyr)',       'steelblue'),
        (ax_nn_mass, 'mass',          'Mass (M☉)',       'seagreen'),
        (ax_nn_ml,   'mixing_length', 'Mixing length α', 'coral'),
    ]:
        ax.set_title(f"NN distribution: {label}", fontsize=10)
        if nn is not None and param in nn.columns:
            vals = nn[param].dropna()
            ax.hist(vals, bins=10, color=color, alpha=0.7, edgecolor='white')
            fit_val = res.get(f'fit_{param}', np.nan)
            if not np.isnan(fit_val):
                ax.axvline(fit_val, color='k', lw=2, label=f'Best fit: {fit_val:.2f}')
                ax.legend(fontsize=8)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel('Count', fontsize=9)

    # Annotate with key results
    result_str = (
        f"Fit loss: {res['fit_loss']:.4f}   "
        f"Converged: {res['fit_converged']}\n"
        f"Age: {res['fit_age']:.2f} Gyr   "
        f"Mass: {res['fit_mass']:.3f} M☉   "
        f"α_MLT: {res['fit_mixing_length']:.3f}"
    )
    fig.text(0.5, 0.01, result_str, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    outpath = f'results/diagnostics/star_{i+1:03d}_{star_id.replace("/","_")}.png'
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    → Age={res['fit_age']:.2f} Gyr  Mass={res['fit_mass']:.3f} M☉  "
          f"α_MLT={res['fit_mixing_length']:.3f}  loss={res['fit_loss']:.4f}")

# ── Save results table ────────────────────────────────────────────────────────
results_df = pd.DataFrame(result_rows)
results_df.to_csv('results/fit_results.csv', index=False)
print(f"\nResults saved to results/fit_results.csv")

# ── Summary plots ─────────────────────────────────────────────────────────────
print("Making summary plots...")
df = results_df[results_df['fit_success']].copy()

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle(f"Platinum Sample — JT2017t11 Fit Summary  (N={len(df)}/{n_stars} successful)",
             fontsize=14, fontweight='bold')
axes = axes.flatten()

# 1. Age histogram
ax = axes[0]
ax.hist(df['fit_age'].dropna(), bins=20, color='steelblue', edgecolor='white')
ax.set_xlabel('Age (Gyr)', fontsize=11)
ax.set_ylabel('N', fontsize=11)
ax.set_title('Age distribution', fontsize=12)

# 2. Mass histogram
ax = axes[1]
ax.hist(df['fit_mass'].dropna(), bins=20, color='seagreen', edgecolor='white')
ax.set_xlabel('Mass (M☉)', fontsize=11)
ax.set_title('Mass distribution', fontsize=12)

# 3. Mixing length histogram
ax = axes[2]
ax.hist(df['fit_mixing_length'].dropna(), bins=20, color='coral', edgecolor='white')
ax.set_xlabel('Mixing length α', fontsize=11)
ax.set_title('Mixing length distribution', fontsize=12)

# 4. HR diagram — coloured by age
ax = axes[3]
sc = ax.scatter(df['teff_obs'], df['lum_obs'], c=df['fit_age'],
                cmap='plasma', s=50, zorder=3)
plt.colorbar(sc, ax=ax, label='Age (Gyr)')
ax.set_xlabel('Teff (K)', fontsize=11)
ax.set_ylabel('log(L/L☉)', fontsize=11)
ax.set_title('H-R diagram (colour = age)', fontsize=12)
ax.invert_xaxis()

# 5. Kiel diagram — coloured by mass
ax = axes[4]
sc = ax.scatter(df['teff_obs'], df['logg_obs'], c=df['fit_mass'],
                cmap='viridis', s=50, zorder=3)
plt.colorbar(sc, ax=ax, label='Mass (M☉)')
ax.set_xlabel('Teff (K)', fontsize=11)
ax.set_ylabel('log g', fontsize=11)
ax.set_title('Kiel diagram (colour = mass)', fontsize=12)
ax.invert_xaxis()
ax.invert_yaxis()

# 6. [M/H] vs age
ax = axes[5]
ax.scatter(df['mh_obs'], df['fit_age'], c='steelblue', s=40, alpha=0.8)
ax.set_xlabel('[M/H]', fontsize=11)
ax.set_ylabel('Age (Gyr)', fontsize=11)
ax.set_title('[M/H] vs Age', fontsize=12)

# 7. Teff residual (obs - fit)
ax = axes[6]
resid_teff = df['teff_obs'] - df['fit_teff']
ax.hist(resid_teff.dropna(), bins=20, color='slateblue', edgecolor='white')
ax.axvline(0, color='k', lw=1.5)
ax.set_xlabel('Teff_obs − Teff_fit (K)', fontsize=11)
ax.set_title('Teff residuals', fontsize=12)

# 8. logg residual
ax = axes[7]
resid_logg = df['logg_obs'] - df['fit_logg']
ax.hist(resid_logg.dropna(), bins=20, color='darkorange', edgecolor='white')
ax.axvline(0, color='k', lw=1.5)
ax.set_xlabel('logg_obs − logg_fit', fontsize=11)
ax.set_title('log g residuals', fontsize=12)

# 9. Fit loss distribution
ax = axes[8]
losses = df['fit_loss'].dropna()
ax.hist(losses, bins=25, color='grey', edgecolor='white')
ax.axvline(FIT_TOL, color='r', lw=1.5, ls='--', label=f'tol={FIT_TOL}')
ax.set_xlabel('Fit loss', fontsize=11)
ax.set_title('Loss distribution (red = tolerance)', fontsize=12)
ax.legend()

plt.tight_layout()
fig.savefig('results/summary_plots.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Summary plots saved to results/summary_plots.png")
print("\nDone.")
