from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "policy_io_diagram.png"


def add_box(ax, x, y, w, h, text, fc, ec="#243042", fontsize=12, dashed=False, weight="regular"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#14202b",
        wrap=True,
    )


def add_arrow(ax, start, end, color="#243042", lw=2.0, dashed=False, connectionstyle="arc3"):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arr)


def add_label(ax, x, y, text, fontsize=14, color="#243042", weight="bold", ha="left"):
    ax.text(x, y, text, fontsize=fontsize, color=color, weight=weight, ha=ha, va="center")


def main():
    fig, ax = plt.subplots(figsize=(18, 10), dpi=200)
    fig.patch.set_facecolor("#f7f3ea")
    ax.set_facecolor("#f7f3ea")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_label(ax, 0.03, 0.965, "Policy Inputs -> Actuator Outputs", fontsize=24, color="#102231")
    ax.text(
        0.03,
        0.93,
        "High-level map of what the policy consumes, how it turns observations into commands, and what finally reaches the plant/actuators.",
        fontsize=12,
        color="#425466",
        ha="left",
        va="center",
    )

    # Section headers
    add_label(ax, 0.06, 0.86, "Inputs Required", fontsize=17, color="#6f4e37")
    add_label(ax, 0.39, 0.86, "Policy Translation", fontsize=17, color="#6f4e37")
    add_label(ax, 0.73, 0.86, "Actuator / Plant Output", fontsize=17, color="#6f4e37")

    # Left column
    add_box(
        ax,
        0.05,
        0.67,
        0.24,
        0.13,
        "Geometry\n\np_def - center\np_att or p_Ak - center\nrel = p_att - p_def",
        fc="#f6d7b0",
        fontsize=12,
    )
    add_box(
        ax,
        0.05,
        0.50,
        0.24,
        0.12,
        "Kinematics\n\nv_def\nv_att or v_Ak",
        fc="#f6d7b0",
        fontsize=12,
    )
    add_box(
        ax,
        0.05,
        0.31,
        0.24,
        0.15,
        "Optional extras\n\nKF / EKF beliefs for opponent state\nfuel fractions\nstudent inputs: sigma_feat + u_prev",
        fc="#d7eadf",
        fontsize=11.5,
        dashed=True,
    )
    add_box(
        ax,
        0.05,
        0.14,
        0.24,
        0.11,
        "1vN attacker note\n\nRL attacker first ego-permutes obs so attacker k becomes slot 0",
        fc="#d9e6f2",
        fontsize=11.5,
        dashed=True,
    )

    # Middle column
    add_box(
        ax,
        0.37,
        0.67,
        0.27,
        0.16,
        "Observation vector\n\n1v1: [p1c, p2c, rel, v1, v2, fuel?]\n1vN: [p1c, pA..., rel..., v1, vA..., fuel?]",
        fc="#fce6bf",
        fontsize=11.8,
    )
    add_box(
        ax,
        0.39,
        0.46,
        0.23,
        0.15,
        "Policy core\n\nActorCriticDiff\nor\nPartialObsStudentPolicy",
        fc="#f8cab9",
        fontsize=12,
    )
    add_box(
        ax,
        0.39,
        0.27,
        0.23,
        0.13,
        "Internal transform\n\noptional prior layer + MLP / recurrent encoder\n-> Normal(mu, std)",
        fc="#f8cab9",
        fontsize=11.3,
    )
    add_box(
        ax,
        0.39,
        0.09,
        0.23,
        0.12,
        "Action squash\n\na_cmd = tanh(u_raw) * umax",
        fc="#f8cab9",
        fontsize=12,
        weight="bold",
    )

    # Right column
    add_box(
        ax,
        0.71,
        0.67,
        0.23,
        0.12,
        "Policy output\n\na_cmd in R^D\n(D=2 or D=3 acceleration command)",
        fc="#d8e9f6",
        fontsize=12,
    )
    add_box(
        ax,
        0.71,
        0.50,
        0.23,
        0.10,
        "Environment interface\n\nclip to [-umax, +umax]",
        fc="#d8e9f6",
        fontsize=12,
    )
    add_box(
        ax,
        0.69,
        0.31,
        0.12,
        0.13,
        "No fuel\n\na_real = a_cmd",
        fc="#dff1da",
        fontsize=12,
    )
    add_box(
        ax,
        0.84,
        0.28,
        0.12,
        0.19,
        "Fuel model\n\nF_req = m * a_cmd\n|F| <= Tmax\na_real = F / m\nmass decreases",
        fc="#ffe0bf",
        fontsize=11.5,
    )
    add_box(
        ax,
        0.71,
        0.08,
        0.23,
        0.12,
        "Plant update\n\nx_(k+1) = Ad x_k + Bd a_real",
        fc="#d8e9f6",
        fontsize=12,
        weight="bold",
    )

    # Main arrows
    add_arrow(ax, (0.29, 0.735), (0.37, 0.735))
    add_arrow(ax, (0.29, 0.56), (0.37, 0.73), dashed=True, connectionstyle="arc3,rad=0.05")
    add_arrow(ax, (0.29, 0.385), (0.39, 0.53), dashed=True, connectionstyle="arc3,rad=0.06")
    add_arrow(ax, (0.29, 0.195), (0.39, 0.53), dashed=True, connectionstyle="arc3,rad=0.08")

    add_arrow(ax, (0.505, 0.67), (0.505, 0.61))
    add_arrow(ax, (0.505, 0.46), (0.505, 0.40))
    add_arrow(ax, (0.505, 0.27), (0.505, 0.21))

    add_arrow(ax, (0.62, 0.15), (0.71, 0.73))
    add_arrow(ax, (0.825, 0.67), (0.825, 0.60))
    add_arrow(ax, (0.77, 0.50), (0.75, 0.44))
    add_arrow(ax, (0.88, 0.50), (0.90, 0.47))
    add_arrow(ax, (0.75, 0.31), (0.79, 0.20))
    add_arrow(ax, (0.90, 0.28), (0.86, 0.20))

    # Small note boxes
    add_box(
        ax,
        0.35,
        0.84,
        0.31,
        0.06,
        "Deterministic eval uses the policy mean. Stochastic rollout samples from the Gaussian before the tanh squash.",
        fc="#f3efe5",
        fontsize=10.5,
        dashed=True,
    )
    add_box(
        ax,
        0.70,
        0.84,
        0.25,
        0.06,
        "If fuel is off, the bounded policy command is the realized actuator acceleration.",
        fc="#f3efe5",
        fontsize=10.5,
        dashed=True,
    )

    # Footer
    ax.text(
        0.03,
        0.03,
        "Code anchors: core/env.py, core_1v2/env.py, rl_infer.py, rl_infer_1v2.py, core/models.py, core/distill.py",
        fontsize=10.5,
        color="#4b5d6b",
        ha="left",
        va="center",
    )

    plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
