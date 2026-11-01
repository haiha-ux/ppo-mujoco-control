"""Learning curves (mean +- std over seeds) and results/summary.md from logs/*.csv and results/eval.json."""
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = np.arange(50_000, 1_000_001, 10_000)
WIN = 50_000
COL = {"ours": "C0", "sb3": "C1", "noobsnorm": "C3", "nogae": "C2"}
LAB = {"ours": "ours", "sb3": "SB3 PPO", "noobsnorm": "ours, no obs. normalisation", "nogae": "ours, lambda=1 (no GAE)"}


def curve(path):
    d = pd.read_csv(path)
    return np.array([d.episode_return[(d.step > g - WIN) & (d.step <= g)].mean() for g in GRID])


def runs():
    r = {}
    for p in glob.glob("logs/*_s*.csv"):
        m = re.match(r"logs/(.+?-v5)_(.+)_s(\d+)\.csv", p)
        r.setdefault((m[1], m[2]), []).append(p)
    return r


def draw(ax, env, impls, R):
    for im in impls:
        if (env, im) not in R: continue
        c = np.stack([curve(p) for p in sorted(R[(env, im)])])
        m, s = np.nanmean(c, 0), np.nanstd(c, 0)
        ax.plot(GRID, m, color=COL[im], label=f"{LAB[im]} (n={len(c)})")
        ax.fill_between(GRID, m - s, m + s, color=COL[im], alpha=0.2)
    ax.set_title(env); ax.set_xlabel("env steps"); ax.set_ylabel("training episode return (raw)"); ax.grid(alpha=.3)


R = runs()
envs = sorted({k[0] for k in R if k[1] in ("ours", "sb3")})
fig, axs = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4), squeeze=False)
for ax, env in zip(axs[0], envs): draw(ax, env, ["ours", "sb3"], R)
axs[0][0].legend(fontsize=8); plt.tight_layout(); plt.savefig("results/curves.png", dpi=110); plt.close()

if ("Hopper-v5", "noobsnorm") in R or ("Hopper-v5", "nogae") in R:
    fig, ax = plt.subplots(figsize=(6, 4))
    draw(ax, "Hopper-v5", ["ours", "noobsnorm", "nogae"], R); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("results/ablation_hopper.png", dpi=110); plt.close()

ev = json.load(open("results/eval.json"))
L = ["| Env | Run | Seeds | Final eval return, mean +- std over seeds | Per-seed means | Train wall time / run | Training steps/s |", "|---|---|---|---|---|---|---|"]
for (env, im) in sorted(R):
    names = sorted(os.path.basename(p)[:-4] for p in R[(env, im)])
    v = [ev[n]["mean"] for n in names if n in ev]
    t = [json.load(open(f"logs/{n}.time.json"))["wall_s"] for n in names]
    if not v: continue
    L.append(f"| {env} | {LAB[im]} | {len(v)} | {np.mean(v):.0f} +- {np.std(v, ddof=1) if len(v) > 1 else 0:.0f} | "
             f"{', '.join(f'{x:.0f}' for x in v)} | {np.mean(t)/60:.1f} min | {1e6/np.mean(t):.0f} |")
open("results/summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
