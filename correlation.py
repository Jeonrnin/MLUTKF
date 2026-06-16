import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import string
from scipy import interpolate
from plot import *

# =====================================================================================
# utility
# =====================================================================================
def calc_corr(x, x_intp, x_axis_shift, y, intp):
    N = len(x)
    y = y.copy()
    if x_axis_shift:
        y[0:N-2] = y[1:N-1]

    if not intp:
        return y

    f_intp = interpolate.interp1d(x, y, kind='cubic')
    return f_intp(x_intp)


# =====================================================================================
# main plotting function (paper version)
# =====================================================================================
def plot_corr_paper(data_dir, plot_show=True, plot_outdir='fig', plot_save=True):

    # --------------------------------------------------
    # axis setup
    # --------------------------------------------------
    N = 40
    x = np.arange(N)
    x_intp = np.arange(0, N-1+0.1, 0.1)

    x_axis_shift = True
    intp = False

    x_axs = x_intp.copy() if intp else x.copy()
    if x_axis_shift:
        x_axs += 1

    import matplotlib.pyplot as plt
    set_plot_font(plt, 0)

    # --------------------------------------------------
    # common
    # --------------------------------------------------
    h_types = [0, 1, 2]
    letters = list(string.ascii_lowercase)

    # --------------------------------------------------
    # reference correlation (black solid line)
    # GETKF, Ne=50 (proxy for "true" correlation)
    # --------------------------------------------------
    ref_corr = []
    for h in h_types:
        fn = os.path.join(
            data_dir,
            f"3_correlation_enkf_method_18_h_type_{h}_nem_50_nobs_100_obs_loc_dist_8.0_y.txt"
        )
        y = np.loadtxt(fn)
        ref_corr.append(calc_corr(x, x_intp, x_axis_shift, y, intp))

    # --------------------------------------------------
    # method definitions (row-wise)
    # --------------------------------------------------
    methods = [
        # row 0: LETKF (local)
        dict(
            name="LETKF",
            localized=False,
            variants=[
                dict(enkf_method=3, nem=12, obs_loc_dist=1.0, label=r'LETKF ($N_e=12$)'),
                dict(enkf_method=3, nem=50, obs_loc_dist=2.0, label=r'LETKF ($N_e=50$)'),
            ],
        ),
        # row 1: LUTKF (local)
        dict(
            name="LUTKF",
            localized=False,
            enkf_method=5,
            nem=4,
            obs_loc_dist=2.0,
            label=r'LUTKF ($N_e=4$)'
        ),
        # row 2: MLUTKF (local)
        dict(
            name="MLUTKF",
            localized=False,
            enkf_method=24,
            nem=12,
            obs_loc_dist=11.0,
            label=r'MLUTKF ($N_e=12$)'
        )
    ]

    # --------------------------------------------------
    # figure
    # --------------------------------------------------
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    for irow, m in enumerate(methods):
        for icol, h in enumerate(h_types):

            # --- reference (black solid) ---
            axs[irow, icol].plot(
                x_axs, ref_corr[icol],
                color='k', linewidth=1.8,
                label=r'Ref. (GETKF, $N_e=50$)'
            )

            # --- method variants ---
            if "variants" in m:
                variants = m["variants"]
            else:
                variants = [dict(
                    enkf_method=m["enkf_method"],
                    nem=m["nem"],
                    obs_loc_dist=m["obs_loc_dist"],
                    label=m["label"]
                )]

            for v in variants:
                fn = os.path.join(
                    data_dir,
                    f"3_correlation_enkf_method_{v['enkf_method']}"
                    f"_h_type_{h}_nem_{v['nem']}_nobs_100"
                    f"_obs_loc_dist_{v['obs_loc_dist']}_y.txt"
                )
                y = np.loadtxt(fn)
                corr = calc_corr(x, x_intp, x_axis_shift, y, intp)

                axs[irow, icol].plot(
                    x_axs, corr,
                    label=v["label"]
                )

            # --- panel label ---
            text = h_mathexp_str(h)
            if m['localized']:
                text += ', localized'

            axs[irow, icol].text(
                0.025, 0.955,
                f"{letters[irow*3 + icol]}) {text}",
                transform=axs[irow, icol].transAxes,
                fontsize=15,
                va="top",
                ha="left",
            )

            # --- cosmetics ---
            axs[irow, icol].set_xlim(0, 40)
            axs[irow, icol].set_ylim(-0.3, 1.1)
            axs[irow, icol].tick_params(direction='in')

            if icol == 0:
                axs[irow, icol].set_ylabel(
                    r"Correlation [$x_{20},\mathbf{x}$]",
                    fontsize=18
                )
            if irow == len(methods) - 1:
                axs[irow, icol].set_xlabel(
                    "Variables", fontsize=18
                )


            legend_kw = dict(
                loc='upper right',
                #bbox_to_anchor=(0.98, 0.98),
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                facecolor='white',
                edgecolor='black',
                fontsize=10.5,
                borderpad=0.35,
                labelspacing=0.25,
                handlelength=1.5,
                handletextpad=0.5,
                borderaxespad=0.6
            )
            axs[irow, icol].legend(**legend_kw)
            # axs[irow, icol].legend(loc='upper right',fontsize=10,frameon=True)

    fig.tight_layout()

    if plot_save:
        save_plot(
            plt,
            os.path.join(plot_outdir, "corr_LET_LUT_MLUT"),
            ["pdf", "svg", "eps"]
        )
    if plot_show:
        plt.show()


# =====================================================================================
# run
# =====================================================================================
if __name__ == "__main__":
    plot_corr_paper(
        data_dir="data",
        plot_show=False,
        plot_outdir="fig",
        plot_save=True
    )
