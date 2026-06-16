from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter, FixedLocator

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import re
import io
import contextlib
import time
import os

from plot import h_mathexp_str
from L96enkf import L96enkf

OUTDIR = "fig"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# 1) Case select
# ----------------------------
# Case 1) Figure 1 / Case A: nobs=40,  errstddev=0.1
# Case 2) Figure 2 / Case B: nobs=100, errstddev=0.1
# Case 3) Figure 3 / Case C: nobs=100, errstddev=1.0
CASE = 1

N = 40
obs_pos_dst = 1
obs_pos_dst_seed = 592

# ----------------------------
# 2) Case-dependent parameters
# ----------------------------
CASE_CONFIG = {
    1: dict(
        case_name="Case A",
        nobs=40,
        errstddev=0.1,
        fig_name="figure1.pdf",

        LETKF_obs_loc=9.0,

        LUTKF_nem_sclfct=2,
        LUTKF_obs_loc=2.0,
        LUTKF_sp_dist=0,

        GETKF_cov_loc=6.0,
        GETKF_thresh=0.5,

        MLUTKF_nem_sclfct=6,
        MLUTKF_obs_loc=9.0,
        MLUTKF_cov_loc=8.0,
        MLUTKF_thresh=0.9,
    ),

    2: dict(
        case_name="Case B",
        nobs=100,
        errstddev=0.1,
        fig_name="figure2.pdf",

        LETKF_obs_loc=8.0,

        LUTKF_nem_sclfct=2,
        LUTKF_obs_loc=2.0,
        LUTKF_sp_dist=0,

        GETKF_cov_loc=3.0,
        GETKF_thresh=0.4,

        MLUTKF_nem_sclfct=6,
        MLUTKF_obs_loc=11.0,
        MLUTKF_cov_loc=4.0,
        MLUTKF_thresh=0.9,
    ),

    3: dict(
        case_name="Case C",
        nobs=100,
        errstddev=1.0,
        fig_name="figure3.pdf",

        LETKF_obs_loc=2.0,

        LUTKF_nem_sclfct=2,
        LUTKF_obs_loc=2.0,
        LUTKF_sp_dist=0,

        GETKF_cov_loc=2.0,
        GETKF_thresh=0.3,

        MLUTKF_nem_sclfct=6,
        MLUTKF_obs_loc=11.0,
        MLUTKF_cov_loc=13.0,
        MLUTKF_thresh=0.8,
    ),
}

if CASE not in CASE_CONFIG:
    raise ValueError("CASE must be 1, 2, or 3")

cfg = CASE_CONFIG[CASE]

case_name = cfg["case_name"]
nobs = cfg["nobs"]
errstddev = cfg["errstddev"]
fig_name = cfg["fig_name"]

LETKF_obs_loc = cfg["LETKF_obs_loc"]

LUTKF_nem_sclfct = cfg["LUTKF_nem_sclfct"]
LUTKF_obs_loc = cfg["LUTKF_obs_loc"]
LUTKF_sp_dist = cfg["LUTKF_sp_dist"]

GETKF_cov_loc = cfg["GETKF_cov_loc"]
GETKF_thresh = cfg["GETKF_thresh"]

MLUTKF_nem_sclfct = cfg["MLUTKF_nem_sclfct"]
MLUTKF_obs_loc = cfg["MLUTKF_obs_loc"]
MLUTKF_cov_loc = cfg["MLUTKF_cov_loc"]
MLUTKF_thresh = cfg["MLUTKF_thresh"]

print(f"Running {case_name}: nobs={nobs}, errstddev={errstddev}")
print(f"Output figure: {fig_name}")

# ----------------------------
# 3) Plot settings
# ----------------------------
LABEL_FS = 16
TICK_FS = 14
PANEL_FS = 15
LEGEND_FS = 9

LETKF_Ne_list = [4, 6, 8, 10, 12, 16, 20, 30, 40, 50, 60, 80, 100]
GETKF_Ne_list = LETKF_Ne_list

h_types = [0, 1, 2]
panel_titles = ["h(x)=x", "h(x)=|x|", "h(x)=ln(|x|)"]


def run_and_get_prior_metrics(**kwargs):
    try:
        obj = L96enkf(**kwargs)
        start = time.time()
        obj.fcst_and_da()

        rmse_b = getattr(obj, "xrmsd_b", None)
        spread_b = getattr(obj, "xstd_b_zmean", None)

        if (rmse_b is None) or (spread_b is None):
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                obj.show_perf(start)
            out = f.getvalue()

            m1 = re.search(r"stddev_b:\s*([0-9eE\.\+\-]+)", out)
            m2 = re.search(r"rmse_b:\s*([0-9eE\.\+\-]+)", out)

            if m1 and m2:
                spread_b = float(m1.group(1))
                rmse_b = float(m2.group(1))
            else:
                return np.nan, np.nan

        return float(rmse_b), float(spread_b)

    except Exception as e:
        print("Run failed, skipped:", e)
        return np.nan, np.nan


def make_kwargs(
    enkf_method,
    h_type,
    nem,
    nem_sclfct=None,
    sp_dist=None,
    obs_loc=None,
    cov_loc=None,
    thresh=None,
):
    kw = dict(
        nts=5000,
        enkf_method=enkf_method,
        nEnKF=1,
        dt_da=0.05,
        N=N,
        nem=nem,
        proc_errstddev=errstddev,
        obs_errstddev=errstddev,
        nobs=nobs,
        obs_pos_dst_seed=obs_pos_dst_seed,
        obs_pos_dst=obs_pos_dst,
        h_type=h_type,
        mpi=False,
        loop_test=True,

        plot_show=False,
        plot_save=False,
        plot_data_save=False,
    )

    if nem_sclfct is not None:
        kw["nem_sclfct"] = nem_sclfct
    if sp_dist is not None:
        kw["sp_dist"] = sp_dist

    if obs_loc is not None:
        kw["obs_loc_dist"] = obs_loc
    if cov_loc is not None:
        kw["cov_loc_dist"] = cov_loc
    if thresh is not None:
        kw["row_rank_thresh"] = thresh

    return kw


# ----------------------------
# 4) Run & plot
# ----------------------------
fig, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=False)

for ax, h_type, title in zip(axes, h_types, panel_titles):

    # LETKF curve
    letkf_rmse = []
    letkf_spread = []

    for Ne in LETKF_Ne_list:
        kw = make_kwargs(
            enkf_method=3,
            h_type=h_type,
            nem=Ne,
            obs_loc=LETKF_obs_loc,
        )
        rmse_b, spread_b = run_and_get_prior_metrics(**kw)
        letkf_rmse.append(rmse_b)
        letkf_spread.append(spread_b)

    # LUTKF fixed Ne=4
    kw = make_kwargs(
        enkf_method=5,
        h_type=h_type,
        nem=1,
        nem_sclfct=LUTKF_nem_sclfct,
        sp_dist=LUTKF_sp_dist,
        obs_loc=LUTKF_obs_loc,
    )
    lutkf_rmse, lutkf_spread = run_and_get_prior_metrics(**kw)

    # GETKF curve
    getkf_rmse_list = []
    getkf_spread_list = []

    for Ne in GETKF_Ne_list:
        kw = make_kwargs(
            enkf_method=18,
            h_type=h_type,
            nem=Ne,
            cov_loc=GETKF_cov_loc,
            thresh=GETKF_thresh,
        )
        rmse_b, spread_b = run_and_get_prior_metrics(**kw)
        getkf_rmse_list.append(rmse_b)
        getkf_spread_list.append(spread_b)

    # MLUTKF fixed Ne=12
    kw = make_kwargs(
        enkf_method=24,
        h_type=h_type,
        nem=1,
        nem_sclfct=MLUTKF_nem_sclfct,
        obs_loc=MLUTKF_obs_loc,
        cov_loc=MLUTKF_cov_loc,
        thresh=MLUTKF_thresh,
    )
    mlutkf_rmse, mlutkf_spread = run_and_get_prior_metrics(**kw)

    LUTKF_Ne = 4
    MLUTKF_Ne = 12

    xmin, xmax = min(LETKF_Ne_list), max(LETKF_Ne_list)

    # LETKF
    ax.plot(LETKF_Ne_list, letkf_rmse, "ko-", label="LETKF RMSE")
    ax.plot(LETKF_Ne_list, letkf_spread, "k--", label="LETKF spread")

    # LUTKF horizontal reference lines
    ax.hlines(
        [lutkf_rmse],
        xmin,
        xmax,
        colors="r",
        linestyles="-",
        label=rf"LUTKF ($N_e={LUTKF_Ne}$) RMSE",
    )
    ax.hlines(
        [lutkf_spread],
        xmin,
        xmax,
        colors="r",
        linestyles="--",
        label=rf"LUTKF ($N_e={LUTKF_Ne}$) spread",
    )

    # GETKF
    xg = np.array(GETKF_Ne_list, float)
    ax.plot(xg, getkf_rmse_list, "bo-", label="GETKF RMSE")
    ax.plot(xg, getkf_spread_list, "b--", label="GETKF spread")

    # MLUTKF horizontal reference lines
    ax.hlines(
        [mlutkf_rmse],
        xmin,
        xmax,
        colors="g",
        linestyles="-",
        label=rf"MLUTKF ($N_e={MLUTKF_Ne}$) RMSE",
    )
    ax.hlines(
        [mlutkf_spread],
        xmin,
        xmax,
        colors="g",
        linestyles="--",
        label=rf"MLUTKF ($N_e={MLUTKF_Ne}$) spread",
    )

    # panel label
    ax.text(
        0.025,
        0.955,
        f"{chr(97 + h_type)}) {h_mathexp_str(h_type)}",
        transform=ax.transAxes,
        fontsize=PANEL_FS,
        va="top",
        ha="left",
    )

    # axis labels / ticks
    ax.set_ylabel("Prior RMSE / spread", fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS, direction="in")
    ax.tick_params(axis="both", which="minor", direction="in")

    # x-axis
    ax.set_xscale("log")
    ax.set_xlim(xmin / 1.05, xmax * 1.05)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=3))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # y-axis
    ax.set_yscale("log")
    ax.set_ylim(ax.get_ylim()[0], min(ax.get_ylim()[1] * 1.15, 9.5))
    ax.yaxis.set_major_locator(FixedLocator([1.0]))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(False)

    leg = ax.legend(
        loc="upper right",
        fontsize=LEGEND_FS,
        ncol=1,
        mode=None,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        borderpad=0.45,
        labelspacing=0.30,
        handlelength=1.8,
        handletextpad=0.55,
        borderaxespad=0.6,
    )
    leg.get_frame().set_linewidth(0.8)


axes[-1].set_xlabel(r"Number of ensemble members, $N_e$", fontsize=LABEL_FS)

plt.tight_layout(rect=[0, 0, 1, 0.97])

outpath = os.path.join(OUTDIR, fig_name)
plt.savefig(outpath, dpi=300, bbox_inches="tight")

print("Saved:", outpath)