"""Plot the two g(tau) noise schedules discussed in posts/flow-grpo-wan.md.

Both values are taken from the Flow-GRPO implementation rather than invented:

  paper form      g(tau) = a * sqrt(tau / (1 - tau)),  a = noise_level = 0.7
                  (config/base.py: `sample.noise_level = 0.7`; the SD3 path in
                  flow_grpo/diffusers_patch/sd3_sde_with_logprob.py computes
                  `sqrt(sigma / (1 - sigma)) * noise_level`)

  Wan2.1 form     g(tau) = sigma_min + (sigma_max - sigma_min) * tau
                  (flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py, with
                  `sigma_max = self.sigmas[1]` and `sigma_min = self.sigmas[-1]`)

sigma_min and sigma_max are therefore not free hyperparameters - they are read off
the UniPC flow-sigma schedule, which this script reconstructs in wan_sigmas().
Note that the Wan path never uses noise_level at all.

Usage:
    python scripts/plot_g_tau.py                      # -> assets/g-tau.svg
    python scripts/plot_g_tau.py --a 0.5 --steps 40
    python scripts/plot_g_tau.py --table              # also print the sampled values

Labels use Unicode rather than matplotlib mathtext on purpose: mathtext emits
Computer Modern font names that no browser has, so the glyphs break in an SVG.
Everything here stays plain text in the site's own webfonts.
"""

import argparse
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Colours lifted from styles.css (hue-73 olive site palette), with the series
# steps lightened/saturated so the two lines separate for colour-blind readers.
INK = "#2d3b0d"          # --ink       hsl(73 76% 20%)
INK_SOFT = "#5c6b3f"     # --ink-soft  hsl(73 28% 34%)
LINE = "#c6cdae"         # --line      hsl(73 20% 78%)
SURFACE = "#f6f8ec"      # --bg        hsl(73 42% 96%)
OLIVE = "#6f8417"        # series 1 - paper schedule
INDIGO = "#5b62c9"       # series 2 - Wan2.1 implementation schedule

SANS = ["Manrope", "Helvetica Neue", "sans-serif"]
MONO = ["IBM Plex Mono", "SFMono-Regular", "monospace"]

# Wan2.1-T2V scheduler/scheduler_config.json, shipped with the Diffusers weights.
NUM_TRAIN_TIMESTEPS = 1000
FLOW_SHIFT = 3.0

# The webfonts above are not installed locally; with svg.fonttype="none" their
# names are written into the SVG and resolved by the browser, so the silent
# local fallback while rendering is expected.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

mpl.rcParams.update({
    "font.family": SANS,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "svg.fonttype": "none",
})


def wan_sigmas(steps, flow_shift=FLOW_SHIFT):
    """The sigma schedule UniPCMultistepScheduler builds when use_flow_sigmas=True.

    Mirrors diffusers' set_timesteps. final_sigmas_type is "zero" in the Wan2.1
    config, so the appended terminal sigma - and hence sigma_min - is exactly 0.
    """
    sigmas = np.linspace(1, 1 / NUM_TRAIN_TIMESTEPS, steps + 1)[:-1]
    sigmas = flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas)
    return np.concatenate([sigmas, [0.0]]).astype(np.float32)


def g_paper(tau, a):
    """Flow-GRPO paper / SD3 path: a * sqrt(tau / (1 - tau)). Diverges as tau -> 1."""
    return a * np.sqrt(tau / (1.0 - tau))


def g_wan(tau, sigma_min, sigma_max):
    """Wan2.1 path: linear interpolation between the schedule's end sigmas."""
    return sigma_min + (sigma_max - sigma_min) * tau


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=float, default=0.7,
                   help="noise_level in the paper schedule (config default 0.7)")
    p.add_argument("--steps", type=int, default=20,
                   help="sample.num_steps, which fixes sigma_max (wan config default 20)")
    p.add_argument("--flow-shift", type=float, default=FLOW_SHIFT,
                   help="scheduler flow_shift (Wan2.1-T2V default 3.0)")
    p.add_argument("--y-max", type=float, default=1.2, help="y-axis cutoff")
    p.add_argument("--out", default="assets/g-tau.svg", help="output path (.svg/.png/.pdf)")
    p.add_argument("--table", action="store_true", help="print a markdown table of values")
    args = p.parse_args()

    sigmas = wan_sigmas(args.steps, args.flow_shift)
    sigma_min, sigma_max = float(sigmas[-1]), float(sigmas[1])
    print(f"steps={args.steps} flow_shift={args.flow_shift:g}  "
          f"sigma_min=sigmas[-1]={sigma_min:.4f}  sigma_max=sigmas[1]={sigma_max:.4f}")

    tau = np.linspace(0.0, 0.9995, 2000)

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)

    ax.plot(tau, g_paper(tau, args.a), color=OLIVE, lw=2.0,
            solid_capstyle="round", zorder=3, label="paper / SD3 path")
    ax.plot(tau, g_wan(tau, sigma_min, sigma_max), color=INDIGO, lw=2.0,
            ls=(0, (6, 4)), dash_capstyle="round", zorder=3, label="Wan2.1 path")

    def label_at(tau_at, fn, color, text, dx, dy, ha):
        """Direct label: colour chip on the curve, text in ink."""
        ax.plot([tau_at], [fn(tau_at)], "o", ms=7, color=color,
                mec=SURFACE, mew=2, zorder=4)
        ax.annotate(text, (tau_at, fn(tau_at)), textcoords="offset points",
                    xytext=(dx, dy), color=INK_SOFT, fontsize=10.5, ha=ha,
                    va="bottom" if dy > 0 else "top", zorder=5)

    # The paper curve sits above the Wan one everywhere, so each label goes on
    # the outer side of its own curve.
    label_at(0.45, lambda t: g_paper(t, args.a), OLIVE,
             f"a·√(τ/(1−τ)),  a = {args.a:g}", -10, 14, "right")
    label_at(0.52, lambda t: g_wan(t, sigma_min, sigma_max), INDIGO,
             "σ_min + (σ_max − σ_min)·τ\n"
             f"σ_min = {sigma_min:.2f},  σ_max = {sigma_max:.2f}", 10, -12, "left")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, args.y_max)
    ax.set_xlabel("flow time  τ     (0 = data,  1 = noise)", fontsize=11, labelpad=8)
    ax.set_ylabel("noise scale  g(τ)", fontsize=11, labelpad=8)
    ax.set_title("Two noise schedules for the Flow-GRPO SDE", fontsize=13,
                 color=INK, loc="left", pad=14)

    ax.set_xticks(np.arange(0.0, 1.01, 0.25))
    ax.set_yticks(np.arange(0.0, args.y_max + 1e-9, 0.2))
    ax.tick_params(labelsize=9.5, length=4, width=0.8, colors=INK_SOFT)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(MONO)

    ax.grid(True, color=LINE, lw=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
        ax.spines[side].set_linewidth(1.0)

    ax.legend(loc="upper left", frameon=False, fontsize=10,
              labelcolor=INK_SOFT, handlelength=2.4).set_zorder(5)

    fig.tight_layout()
    fig.savefig(args.out, transparent=False)
    print(f"wrote {args.out}")

    if args.table:
        print("\n| tau | paper | Wan2.1 |")
        print("| --- | --- | --- |")
        for t in np.arange(0.0, 1.0, 0.1):
            print(f"| {t:.1f} | {g_paper(t, args.a):.3f} | "
                  f"{g_wan(t, sigma_min, sigma_max):.3f} |")


if __name__ == "__main__":
    main()
