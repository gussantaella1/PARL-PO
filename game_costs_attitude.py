# game_costs_attitude.py
# Cost builders for the nonlinear attitude (quaternion) planner/game.

from __future__ import annotations
from typing import Callable, Dict, Any

# --------------------------- Public API ---------------------------------

def build_costs(mode: str = "defender_track", **weights) -> Callable:
    """
    Return a callable cost_fun(m, k, T, ctx) -> (stage_cost_expr, smooth_cost_expr)
    that the attitude PATH model will call at each stage k.

    mode:
      - "defender_track"  : minimize 1 - d·b_x(q)  (point camera at target)
      - "evader_lookaway" : minimize d·b_x(q)      (push boresight away)
      - "neutral"         : effort-only regularization (no pointing term)

    weights (defaults shown):
      w_track=50.0         # pointing/anti-pointing weight
      w_w=1e-2             # |ω|^2
      w_tau=1e-3           # |τ|^2
      w_dw=1e-2            # |ω_{k+1} - ω_k|^2
      w_dtau=1e-4          # |τ_{k+1} - τ_k|^2
      w_spin=0.0           # (optional) spin about boresight penalty
      use_tau_smooth=True  # include τ-smoothness
      use_w_smooth=True    # include ω-smoothness
      cone=None            # optional {'cos_half': float} soft cone (adds hinge)

    The model provides ctx with:
      ctx['d_seq'][k]       : desired world direction (3,)
      ctx['bx_of_q'](m,k)   : returns tuple (bx_x, bx_y, bx_z)
      ctx['omega_bx'](m,k)  : scalar ≈ ω along boresight (optional term)
    """
    # defaults
    W = {
        "w_track": 50.0,
        "w_w": 1e-2,
        "w_tau": 1e-3,
        "w_dw": 1e-2,
        "w_dtau": 1e-4,
        "w_spin": 0.0,
        "use_tau_smooth": True,
        "use_w_smooth": True,
        "cone": None,  # e.g., {"cos_half": 0.965925826}  # cos(15deg)
    }
    W.update(weights or {})

    mode = (mode or "defender_track").lower()
    if mode not in ("defender_track", "evader_lookaway", "neutral"):
        raise ValueError(f"Unknown attitude cost mode: {mode}")

    def cost_fun(m, k: int, T: int, ctx: Dict[str, Any]):
        bx = ctx["bx_of_q"](m, k)            # (bx_x, bx_y, bx_z) Pyomo expr tuple
        d  = ctx["d_seq"][k]                 # numpy (3,)
        # cosine between d and boresight
        cos_align = d[0]*bx[0] + d[1]*bx[1] + d[2]*bx[2]

        # --- pointing term ---
        track_term = 0.0
        if mode == "defender_track":
            # 1 - cos -> 0 at perfect alignment
            track_term += W["w_track"] * (1.0 - cos_align)
        elif mode == "evader_lookaway":
            # minimize cos (push away)
            track_term += W["w_track"] * (cos_align)

        # Optional soft FOV cone hinge: max(0, cos_half - cos_align)^2
        if W["cone"] is not None and "cos_half" in W["cone"]:
            # Smooth hinge via squared positive part using (x + |x|)/2
            cos_half = float(W["cone"]["cos_half"])
            x = (cos_half - cos_align)
            # (x)_+^2 = ((x + |x|)/2)^2 ; use abs() which Pyomo handles via piecewise
            # If you prefer pure-smooth, replace with softplus approx.
            track_term += (x + abs(x))**2 / 4.0

        # --- effort terms ---
        stage_effort = W["w_w"] * sum(m.w[k,i]**2 for i in range(3))
        if k < T-1:
            stage_effort += W["w_tau"] * sum(m.tau[k,i]**2 for i in range(3))

        # --- boresight spin (optional) ---
        spin_term = 0.0
        if W["w_spin"] > 0.0 and "omega_bx" in ctx:
            spin_term = W["w_spin"] * (ctx["omega_bx"](m, k)**2)

        stage_cost = track_term + stage_effort + spin_term

        # --- smoothness (couples k to k+1, so returned separately) ---
        smooth_cost = 0.0
        if k < T-1:
            if W["use_w_smooth"]:
                smooth_cost += W["w_dw"] * sum((m.w[k+1,i] - m.w[k,i])**2 for i in range(3))
            if W["use_tau_smooth"]:
                smooth_cost += W["w_dtau"] * sum((m.tau[k+1,i] - m.tau[k,i])**2 for i in range(3))

        return stage_cost, smooth_cost

    return cost_fun
