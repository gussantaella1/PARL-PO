# game_costs_attitude.py
from __future__ import annotations
from typing import Callable, Dict, Any

__all__ = ["build_costs_attitude", "build_costs"]

def _stage_cost_factory(mode: str, **W) -> Callable:
    """
    Returns cost_fun(m,k,T,ctx)->(stage_expr, smooth_expr) for PATH/Pyomo.
    ctx provides:
      - ctx['d_seq'][k] : np.array(3,)
      - ctx['bx_of_q'](m,k) -> tuple(PyomoExpr,PyomoExpr,PyomoExpr)
      - ctx['omega_bx'](m,k) -> PyomoExpr
    """
    mode = (mode or "defender_track").lower()
    assert mode in ("defender_track", "evader_lookaway", "neutral")

    # defaults
    Wfull = dict(
        w_track=50.0,
        w_w=1e-2,        # |ω|^2
        w_tau=1e-3,      # |τ|^2
        w_dw=1e-2,       # |ω_{k+1}-ω_k|^2
        w_dtau=1e-4,     # |τ_{k+1}-τ_k|^2  (guarded to k<=T-3)
        w_spin=0.0,      # (ω about boresight)^2
        use_w_smooth=True,
        use_tau_smooth=True,
        cone=None,       # {"cos_half": float}  (optional; off by default)
    )
    Wfull.update(W or {})

    def cost_fun(m, k: int, T: int, ctx: Dict[str, Any]):
        bx = ctx["bx_of_q"](m, k)
        d  = ctx["d_seq"][k]
        cos_align = d[0]*bx[0] + d[1]*bx[1] + d[2]*bx[2]

        # pointing term
        if mode == "defender_track":
            track_term = Wfull["w_track"] * (1.0 - cos_align)
        elif mode == "evader_lookaway":
            track_term = Wfull["w_track"] * (cos_align)
        else:
            track_term = 0.0

        # (optional) soft FOV cone — left off unless you tune it explicitly
        if Wfull["cone"] and ("cos_half" in Wfull["cone"]):
            # smooth approximation of hinge: (softplus(cos_half - cos_align))^2
            # implement a C1-ish surrogate: z = cos_half - cos_align
            # NOTE: avoid abs() / piecewise for PATH robustness
            z = Wfull["cone"]["cos_half"] - cos_align
            track_term += z*z  # simplest smooth surrogate; tighten later if needed

        # effort
        stage_effort = Wfull["w_w"]  * sum(m.w[k,i]**2   for i in range(3))
        if k < T-1:
            stage_effort += Wfull["w_tau"] * sum(m.tau[k,i]**2 for i in range(3))

        # boresight spin
        spin_term = 0.0
        if Wfull["w_spin"] > 0.0 and ("omega_bx" in ctx):
            spin_term = Wfull["w_spin"] * (ctx["omega_bx"](m, k)**2)

        stage_cost = track_term + stage_effort + spin_term

        # smoothness across time
        smooth_cost = 0.0
        if k < T-1 and Wfull["use_w_smooth"]:
            smooth_cost += Wfull["w_dw"] * sum((m.w[k+1,i]-m.w[k,i])**2 for i in range(3))
        if k < T-2 and Wfull["use_tau_smooth"]:
            smooth_cost += Wfull["w_dtau"] * sum((m.tau[k+1,i]-m.tau[k,i])**2 for i in range(3))

        return stage_cost, smooth_cost

    return cost_fun

def build_costs_attitude(T: int, cfg: dict):
    """
    Centralized ‘translation-style’ entrypoint for attitude, but returning
    Pyomo-native cost callables for PATH/MCP.

    Reads:
      cfg['att']['mode1'|'mode2']  e.g. "defender_track", "evader_lookaway"
      cfg['att_cost']              weight dict (w_track, w_w, ...)
    Returns:
      (cost1, cost2) — each is cost_fun(m,k,T,ctx)->(stage,smooth)
    """
    att      = dict(cfg.get("att", {}))
    att_cost = dict(cfg.get("att_cost", {}))
    cost1 = _stage_cost_factory(att.get("mode1", "defender_track"), **att_cost)
    cost2 = _stage_cost_factory(att.get("mode2", "defender_track"), **att_cost)
    return cost1, cost2

# Backwards-compatible helper if you prefer the earlier simple API:
def build_costs(mode: str = "defender_track", **weights) -> Callable:
    return _stage_cost_factory(mode, **weights)
