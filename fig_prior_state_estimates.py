import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import string

from L96enkf import L96enkf

OUTDIR = "fig"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# 1) Case select
# ----------------------------
# Case 1) Case A: nobs=40,  h_type=2, errstddev=0.1
# Case 2) Case B: nobs=100, h_type=2, errstddev=0.1
# Case 3) Case C: nobs=100, h_type=2, errstddev=1.0
CASE = 3

h_type = 2
N = 40

CASE_CONFIG = {
    1: dict(
        case_name="Case A",
        nobs=40,
        errstddev=0.1,
        fig_name="figure10.pdf",
        rmse_ylim=(0, 5.2),
        rmse_yticks=[0, 1, 2, 3, 4, 5],
        methods=[
            dict(name="LETKF ($N_e=12$)", enkf_method=3, nem=12, obs_loc=9.0),
            dict(name="LETKF ($N_e=50$)", enkf_method=3, nem=50, obs_loc=9.0),

            dict(name="LUTKF ($N_e=4$)", enkf_method=5, nem=1,
                 nem_sclfct=2, sp_dist=0, obs_loc=2.0),

            dict(name="GETKF ($N_e=50$)", enkf_method=18, nem=50,
                 cov_loc=6.0, thresh=0.5),

            dict(name="MLUTKF ($N_e=12$)", enkf_method=24, nem=1,
                 nem_sclfct=6, obs_loc=9.0, cov_loc=8.0, thresh=0.9),
        ],
    ),

    2: dict(
        case_name="Case B",
        nobs=100,
        errstddev=0.1,
        fig_name="figure11.pdf",
        rmse_ylim=(0, 1),
        rmse_yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        methods=[
            dict(name="LETKF ($N_e=12$)", enkf_method=3, nem=12, obs_loc=3.0),
            dict(name="LETKF ($N_e=50$)", enkf_method=3, nem=50, obs_loc=8.0),

            dict(name="LUTKF ($N_e=4$)", enkf_method=5, nem=1,
                 nem_sclfct=2, sp_dist=0, obs_loc=2.0),

            dict(name="GETKF ($N_e=50$)", enkf_method=18, nem=50,
                 cov_loc=3.0, thresh=0.4),

            dict(name="MLUTKF ($N_e=12$)", enkf_method=24, nem=1,
                 nem_sclfct=6, obs_loc=11.0, cov_loc=4.0, thresh=0.9),
        ],
    ),

    3: dict(
        case_name="Case C",
        nobs=100,
        errstddev=1.0,
        fig_name="figure12.pdf",
        rmse_ylim=(0, 4.2),
        rmse_yticks=[0, 1, 2, 3, 4],
        methods=[
            dict(name="LETKF ($N_e=12$)", enkf_method=3, nem=12, obs_loc=1.0),
            dict(name="LETKF ($N_e=50$)", enkf_method=3, nem=50, obs_loc=2.0),

            dict(name="LUTKF ($N_e=4$)", enkf_method=5, nem=1,
                 nem_sclfct=2, sp_dist=0, obs_loc=2.0),

            dict(name="GETKF ($N_e=50$)", enkf_method=18, nem=50,
                 cov_loc=2.0, thresh=0.3),

            dict(name="MLUTKF ($N_e=12$)", enkf_method=24, nem=1,
                 nem_sclfct=6, obs_loc=11.0, cov_loc=13.0, thresh=0.8),
        ],
    ),
}

if CASE not in CASE_CONFIG:
    raise ValueError("CASE must be 1, 2, or 3")

cfg = CASE_CONFIG[CASE]

case_name = cfg["case_name"]
nobs = cfg["nobs"]
errstddev = cfg["errstddev"]
methods = cfg["methods"]
fig_name = cfg["fig_name"]
rmse_ylim = cfg["rmse_ylim"]
rmse_yticks = cfg["rmse_yticks"]

print(f"Running {case_name}: nobs={nobs}, h_type={h_type}, errstddev={errstddev}")

# ----------------------------
# 2) Fig time window
# ----------------------------
t0, t1 = 3000, 4000     # inclusive/exclusive slicing

# ----------------------------
# 3) Color map
# ----------------------------
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_map = {m["name"]: palette[i % len(palette)] for i, m in enumerate(methods)}


def make_kwargs(m):
    kw = dict(
        nts=5000,
        enkf_method=m["enkf_method"],
        nEnKF=1,
        dt_da=0.05,
        N=N,
        nem=m["nem"],
        proc_errstddev=errstddev,
        obs_errstddev=errstddev,
        nobs=nobs,
        obs_pos_dst_seed=592,
        obs_pos_dst=1,
        h_type=h_type,

        plot_show=False,
        plot_save=False,
        plot_data_save=False,
        mpi=False,
        loop_test=True,
    )

    if "nem_sclfct" in m:
        kw["nem_sclfct"] = m["nem_sclfct"]
    if "sp_dist" in m:
        kw["sp_dist"] = m["sp_dist"]

    if "obs_loc" in m:
        kw["obs_loc_dist"] = m["obs_loc"]
    if "cov_loc" in m:
        kw["cov_loc_dist"] = m["cov_loc"]
    if "thresh" in m:
        kw["row_rank_thresh"] = m["thresh"]

    return kw


def run_one(m):
    obj = L96enkf(**make_kwargs(m))
    start = time.time()
    obj.fcst_and_da()

    print("\n--------------------------------------------")
    print(f"[{m['name']}]")
    obj.show_perf(start)
    print("--------------------------------------------\n")

    xb = np.array(getattr(obj, "xmean_b_acc"))   # (time, N)
    xt = np.array(getattr(obj, "x_truth_acc"))   # (time, N)

    if xb.ndim != 2 or xt.ndim != 2:
        raise RuntimeError("xmean_b_acc / x_truth_acc shape unexpected")

    xb_w = xb[t0:t1, :]
    xt_w = xt[t0:t1, :]

    # Domain-averaged prior state estimate and truth
    x_est = xb_w.mean(axis=1)
    x_true = xt_w.mean(axis=1)

    # Domain-averaged prior RMSE over all 40 variables
    rmse_ts = np.sqrt(np.mean((xb_w - xt_w) ** 2, axis=1))

    return x_true, x_est, rmse_ts


# ----------------------------
# 4) Run
# ----------------------------
results = []

for m in methods:
    x_true, x_est, rmse_ts = run_one(m)
    results.append((m["name"], x_true, x_est, rmse_ts))

T = np.arange(t0, t1)

# ----------------------------
# 5) Plot
# ----------------------------
n_state = len(methods)
fig, axes = plt.subplots(n_state + 1, 1, figsize=(8, 11), sharex=False)

letters = list(string.ascii_lowercase)

# --- state panels ---
for i, (name, x_true, x_est, _) in enumerate(results):
    ax = axes[i]

    ax.plot(T, x_true, "k-", linewidth=1.0, label="True")
    ax.plot(T, x_est, "--", linewidth=1.0, color=color_map[name], label=name)

    ax.set_ylabel("x")

    ax.set_ylim(1.1, 3.6)
    ax.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5])

    ax.tick_params(labelleft=True, labelbottom=True)
    ax.grid(False)

    leg = ax.legend(
        loc="upper right",
        fontsize=10,
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

    # panel label
    ax.text(
        0.017, 0.955, f"{letters[i]})",
        transform=ax.transAxes,
        fontsize=15,
        va="top",
        ha="left",
    )

# --- RMSE panel ---
ax = axes[-1]

for name, _, __, rmse_ts in results:
    ax.plot(T, rmse_ts, linewidth=1.0, color=color_map[name], label=name)

ax.set_ylabel("Prior RMSE")
ax.set_ylim(*rmse_ylim)
ax.set_yticks(rmse_yticks)

ax.tick_params(labelleft=True, labelbottom=True)
ax.grid(False)

leg = ax.legend(
    loc="upper right",
    fontsize=9,
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

ax.text(
    0.017, 0.955, f"{letters[len(axes)-1]})",
    transform=ax.transAxes,
    fontsize=15,
    va="top",
    ha="left",
)

fig.text(0.5, -0.01, "Time steps", ha="center")

plt.tight_layout()

# ----------------------------
# 6) Save
# ----------------------------
outpath = os.path.join(OUTDIR, fig_name)
plt.savefig(outpath, dpi=300, bbox_inches="tight")

print("Saved:", outpath)