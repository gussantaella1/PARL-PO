# game_costs.py
from __future__ import annotations
from typing import Tuple, Callable, Dict, Any
from pyomo.environ import sqrt, exp

# ---- shared small helpers (Pyomo-friendly) ----
def _vec(model_array, k, idxs):
    return [model_array[k, i] for i in idxs]

def _pos(x, k, D):  # [px,py,pz]
    return _vec(x, k, range(D))

def _vel(x, k, D):  # [vx,vy,vz]
    return _vec(x, k, range(D, 2*D))

def _dot(a, b):
    return sum(ai*bi for ai,bi in zip(a,b))

def _sumsq(v):
    return sum(vi*vi for vi in v)

def _norm(v):
    # small epsilon is okay; keep it symbolic-safe
    return sqrt(_sumsq(v) + 1e-9)


def build_game_costs(
    kind: str,
    cfg: Dict[str, Any],
    D: int,
    T: int,
) -> Tuple[Callable, Callable, Callable, Callable]:
    """
    Returns four callables that produce Pyomo expressions:

      l1_k(m,k), l2_k(m,k)  for 0 <= k < T
      l1_T(m),   l2_T(m)    at terminal time T

    Each callable must reference vars on the given model m.
    """

    # ---- bind module-level helpers once; use these aliases everywhere below ----
    POS, VEL, DOT, SUMSQ, NORM = _pos, _vel, _dot, _sumsq, _norm

    k = (kind or "chase_escape_tail").lower()
    cfg = cfg or {}

    # ===================== chase_escape_tail =====================
    # if k == "chase_escape_tail":
    #     c_eff1  = float(cfg.get("effort_w1",   0.01))
    #     c_eff2  = float(cfg.get("effort_w2",   0.01))
    #     c_wfar  = float(cfg.get("w_far",       1.0))
    #     c_wterm = float(cfg.get("w_term",      1.0))
    #     c_ddes  = float(cfg.get("follow_gap",  0.6))
    #     c_wlong = float(cfg.get("w_tail_long", 1.0))
    #     c_wlat  = float(cfg.get("w_tail_lat",  8.0))
    #     c_vref  = float(cfg.get("tail_v_ref",  0.2))
    #     eps_R   = float(cfg.get("path_eps_R",  1e-3))

    #     def tail_cost(pL, pF, vL):
    #         speed = _norm(vL)
    #         vhat = [vi / (speed + 1e-9) for vi in vL]
    #         r = [pF[i] - pL[i] for i in range(D)]
    #         s = _dot(r, vhat)               # parallel component
    #         rperp = [r[i] - s*vhat[i] for i in range(D)]
    #         tail = c_wlong*(s + c_ddes)**2 + c_wlat*_sumsq(rperp)
    #         blend = (speed*speed) / (speed*speed + c_vref*c_vref)
    #         return blend*tail + (1.0 - blend)*_sumsq(r)

    #     def l1_k(m, k_):
    #         p2 = _pos(m.x2, k_, D); v2 = _vel(m.x2, k_, D); p1 = _pos(m.x1, k_, D)
    #         u1 = [m.u1[k_, j] for j in m.U]
    #         return tail_cost(p2, p1, v2) + (c_eff1 + eps_R)*_sumsq(u1)

    #     def l2_k(m, k_):
    #         p1 = _pos(m.x1, k_, D); p2 = _pos(m.x2, k_, D)
    #         u2 = [m.u2[k_, j] for j in m.U]
    #         r  = [p2[i]-p1[i] for i in range(D)]
    #         return -c_wfar*_sumsq(r) + (c_eff2 + eps_R)*_sumsq(u2)

    #     def l1_T(m):
    #         p2 = _pos(m.x2, T, D); v2 = _vel(m.x2, T, D); p1 = _pos(m.x1, T, D)
    #         return c_wterm*tail_cost(p2, p1, v2)

    #     def l2_T(m):
    #         p1 = _pos(m.x1, T, D); p2 = _pos(m.x2, T, D)
    #         r  = [p2[i]-p1[i] for i in range(D)]
    #         return -c_wterm*_sumsq(r)

    #     return l1_k, l2_k, l1_T, l2_T

    # ===================== rendezvous_track =====================
    
    # elif k == "guard_center":
    #     # Player 2: reach the center c
    #     # Player 1: stay close to Player 2 (optional screening alignment)
    #     ar = (cfg.get("arena") or {})
    #     c = [float(ar.get(key, 0.0)) for key in (["cx","cy"] if D == 2 else ["cx","cy","cz"])]

    #     # weights
    #     w2_c      = float(cfg.get("w2_center",    1.0))    # P2 stage: dist to center
    #     w2_c_T    = float(cfg.get("w2_center_T",  20.0))   # P2 terminal
    #     w1_close  = float(cfg.get("w1_close",     2.0))    # P1 stage: closeness to P2
    #     w1_close_T= float(cfg.get("w1_close_T",   20.0))   # P1 terminal
    #     w1_align  = float(cfg.get("w1_align",     0.0))    # optional screening (stage)
    #     w1_align_T= float(cfg.get("w1_align_T",   0.0))    # optional screening (terminal)

    #     # effort regularization
    #     c_eff1 = float(cfg.get("effort_w1", 1e-2))
    #     c_eff2 = float(cfg.get("effort_w2", 1e-2))
    #     eps    = float(cfg.get("eps_cost", 1e-9))

    #     def _align_penalty(p1, p2):
    #         # Encourage (p1 - p2) to align with (c - p2); 0 when perfectly aligned.
    #         v1 = [p1[i] - p2[i] for i in range(D)]
    #         v2 = [(c[i] if i < len(c) else 0.0) - p2[i] for i in range(D)]
    #         n1 = _norm(v1); n2 = _norm(v2)
    #         cos = _dot(v1, v2) / (n1*n2 + eps)
    #         return (1.0 - cos) * (1.0 - cos)

    #     # ---- stage costs ----
    #     def l1_k(m, k_):
    #         p1 = _pos(m.x1, k_, D); p2 = _pos(m.x2, k_, D)
    #         u1 = [m.u1[k_, j] for j in m.U]
    #         stay_close = _sumsq([p1[i] - p2[i] for i in range(D)])
    #         align = _align_penalty(p1, p2) if w1_align > 0.0 else 0.0
    #         stage_cost = w1_close*stay_close + w1_align*align + c_eff1*_sumsq(u1)
    #         return stage_cost

    #     def l2_k(m, k_):
    #         p2 = _pos(m.x2, k_, D)
    #         u2 = [m.u2[k_, j] for j in m.U]
    #         to_center = _sumsq([p2[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
    #         stage_cost = w2_c*to_center + c_eff2*_sumsq(u2)
    #         # raise("Debug")
    #         return stage_cost

    #     # ---- terminal costs ----
    #     def l1_T(m):
    #         p1T = _pos(m.x1, T, D); p2T = _pos(m.x2, T, D)
    #         stay_close_T = _sumsq([p1T[i] - p2T[i] for i in range(D)])
    #         align_T = _align_penalty(p1T, p2T) if w1_align_T > 0.0 else 0.0
    #         return w1_close_T*stay_close_T + w1_align_T*align_T

    #     def l2_T(m):
    #         p2T = _pos(m.x2, T, D)
    #         to_center_T = _sumsq([p2T[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
    #         return w2_c_T*to_center_T

    #     return l1_k, l2_k, l1_T, l2_T
    
    # elif k == "guard_center_v2":
    #     # Player 2: reach the center c
    #     # Player 1: keep the attacker far from the center at terminal time
    #     ar = (cfg.get("arena") or {})
    #     c = [float(ar.get(key, 0.0)) for key in (["cx","cy"] if D == 2 else ["cx","cy","cz"])]

    #     # weights
    #     w2_c       = float(cfg.get("w2_center",    1.0))   # P2 stage: dist to center
    #     w2_c_T     = float(cfg.get("w2_center_T",  20.0))  # P2 terminal
    #     w1_close   = float(cfg.get("w1_close",     2.0))   # P1 stage: closeness to P2
    #     w1_align   = float(cfg.get("w1_align",     0.0))   # optional screening (stage)
    #     c_eff1     = float(cfg.get("effort_w1",    1e-2))
    #     c_eff2     = float(cfg.get("effort_w2",    1e-2))
    #     eps        = float(cfg.get("eps_cost",     1e-9))

    #     # NEW: defender terminal weight for pushing attacker away from center
    #     w1_far_T   = float(cfg.get("w1_far_T",     20.0))

    #     def _align_penalty(p1, p2):
    #         # Encourage (p1 - p2) to align with (c - p2); 0 when perfectly aligned.
    #         v1 = [p1[i] - p2[i] for i in range(D)]
    #         v2 = [(c[i] if i < len(c) else 0.0) - p2[i] for i in range(D)]
    #         n1 = _norm(v1); n2 = _norm(v2)
    #         cos = _dot(v1, v2) / (n1*n2 + eps)
    #         return (1.0 - cos) * (1.0 - cos)

    #     # ---- stage costs (unchanged) ----
    #     def l1_k(m, k_):
    #         p1 = _pos(m.x1, k_, D); p2 = _pos(m.x2, k_, D)
    #         u1 = [m.u1[k_, j] for j in m.U]
    #         stay_close = _sumsq([p1[i] - p2[i] for i in range(D)])
    #         align = _align_penalty(p1, p2) if w1_align > 0.0 else 0.0
    #         return w1_close*stay_close + w1_align*align + c_eff1*_sumsq(u1)

    #     def l2_k(m, k_):
    #         p2 = _pos(m.x2, k_, D)
    #         u2 = [m.u2[k_, j] for j in m.U]
    #         to_center = _sumsq([p2[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
    #         return w2_c*to_center + c_eff2*_sumsq(u2)

    #     # ---- terminal costs (CHANGED) ----
    #     def l1_T(m):
    #         # Defender minimizes negative distance^2 so it *maximizes* attacker distance to center
    #         p2T = _pos(m.x2, T, D)
    #         to_center_T = _sumsq([p2T[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
    #         return -w1_far_T * to_center_T

    #     def l2_T(m):
    #         # Attacker minimizes distance to center
    #         p2T = _pos(m.x2, T, D)
    #         to_center_T = _sumsq([p2T[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
    #         return  w2_c_T * to_center_T

    #     return l1_k, l2_k, l1_T, l2_T
    
    if k == "guard_center_v3":
        # Player 2: reach the center c, but avoid getting too close to Player 1
        # Player 1: keep the attacker far from the center at terminal time
        ar = (cfg.get("arena") or {})
        c = [float(ar.get(key, 0.0)) for key in (["cx","cy"] if D == 2 else ["cx","cy","cz"])]

        # weights (existing)
        w2_c     = float(cfg.get("w2_center",    1.0))    # P2 stage: dist to center
        w2_c_T   = float(cfg.get("w2_center_T",  20.0))   # P2 terminal
        w1_close = float(cfg.get("w1_close",     2.0))    # P1 stage: closeness to P2
        w1_align = float(cfg.get("w1_align",     0.0))    # optional screening (stage)
        c_eff1   = float(cfg.get("effort_w1",    1e-2))
        c_eff2   = float(cfg.get("effort_w2",    1e-2))
        eps      = float(cfg.get("eps_cost",     1e-9))

        # defender terminal weight (existing behavior)
        w1_far_T = float(cfg.get("w1_far_T",     20.0))

        # NEW: attacker “personal space” parameters
        #   d_safe     : preferred minimum distance from defender
        #   w2_avoid   : stage weight for staying outside d_safe
        #   w2_avoid_T : terminal weight (usually smaller than stage or zero)
        d_safe     = float(cfg.get("att_d_safe",   2.0))
        w2_avoid   = float(cfg.get("w2_avoid",     4.0))
        w2_avoid_T = float(cfg.get("w2_avoid_T",   1.0))

        def _align_penalty(p1, p2):
            v1 = [p1[i] - p2[i] for i in range(D)]
            v2 = [(c[i] if i < len(c) else 0.0) - p2[i] for i in range(D)]
            n1 = _norm(v1); n2 = _norm(v2)
            cos = _dot(v1, v2) / (n1*n2 + eps)
            return (1.0 - cos) * (1.0 - cos)

        # smooth positive part: ~max(0, t); differentiable for Pyomo
        def _pospart(t):
            return 0.5*(t + sqrt(t*t + 1e-9))

        # ---- stage costs ----
        def l1_k(m, k_):
            p1 = _pos(m.x1, k_, D); p2 = _pos(m.x2, k_, D)
            u1 = [m.u1[k_, j] for j in m.U]
            stay_close = _sumsq([p1[i] - p2[i] for i in range(D)])
            align = _align_penalty(p1, p2) if w1_align > 0.0 else 0.0
            return w1_close*stay_close + w1_align*align + c_eff1*_sumsq(u1)

        def l2_k(m, k_):
            p1 = _pos(m.x1, k_, D); p2 = _pos(m.x2, k_, D)
            u2 = [m.u2[k_, j] for j in m.U]

            # drive to center (as before)
            to_center = _sumsq([p2[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])

            # NEW: proximity penalty — kicks in only when ||p2 - p1|| < d_safe
            d12 = _norm([p2[i] - p1[i] for i in range(D)])
            avoid = _pospart(d_safe - d12)**2    # = 0 if d12 >= d_safe

            return w2_c*to_center + w2_avoid*avoid + c_eff2*_sumsq(u2)

        # ---- terminal costs ----
        def l1_T(m):
            # Defender: maximize attacker distance to center (minimize negative dist^2)
            p2T = _pos(m.x2, T, D)
            to_center_T = _sumsq([p2T[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
            return -w1_far_T * to_center_T

        def l2_T(m):
            # Attacker: still prefers ending near center, but also not on top of defender
            p1T = _pos(m.x1, T, D); p2T = _pos(m.x2, T, D)
            to_center_T = _sumsq([p2T[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
            d12T = _norm([p2T[i] - p1T[i] for i in range(D)])
            avoidT = _pospart(d_safe - d12T)**2
            return w2_c_T*to_center_T + w2_avoid_T*avoidT

        return l1_k, l2_k, l1_T, l2_T
    
    # game_costs.py
    elif k == "terminal_only":
        ar = (cfg.get("arena") or {})
        c  = [float(ar.get(k, 0.0)) for k in (["cx","cy"] if D==2 else ["cx","cy","cz"])]
        eps = float(cfg.get("eps_cost", 1e-9))

        def POS(x,k): return [x[k,i] for i in range(D)]
        def SUMSQ(v): return sum(vi*vi for vi in v)
        def to_center_sq(p): return SUMSQ([p[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])

        # No stage shaping: let RL/policies decide behavior
        def l1_k(m, k_): return 0.0
        def l2_k(m, k_): return 0.0

        # Fixed terminal goals only
        def l1_T(m):
            p2T = POS(m.x2, m.Kx.last())
            return -to_center_sq(p2T)  # defender maximizes distance

        def l2_T(m):
            p2T = POS(m.x2, m.Kx.last())
            return  to_center_sq(p2T)  # attacker minimizes distance

        return l1_k, l2_k, l1_T, l2_T

    
    # elif k == "guard_center_v3":
    #     ar = (cfg.get("arena") or {})
    #     c = [float(ar.get(key, 0.0)) for key in (["cx","cy"] if D == 2 else ["cx","cy","cz"])]

    #     w2_c     = float(cfg.get("w2_center",    1.0))
    #     w2_c_T   = float(cfg.get("w2_center_T",  20.0))
    #     w1_close = float(cfg.get("w1_close",     2.0))
    #     w1_align = float(cfg.get("w1_align",     0.0))
    #     c_eff1   = float(cfg.get("effort_w1",    1e-2))
    #     c_eff2   = float(cfg.get("effort_w2",    1e-2))
    #     eps      = float(cfg.get("eps_cost",     1e-9))

    #     w1_far_T = float(cfg.get("w1_far_T",     20.0))

    #     d_safe     = float(cfg.get("att_d_safe",   2.0))
    #     w2_avoid   = float(cfg.get("w2_avoid",     4.0))
    #     w2_avoid_T = float(cfg.get("w2_avoid_T",   1.0))

    #     def _align_penalty(p1, p2):
    #         v1 = [p1[i] - p2[i] for i in range(D)]
    #         v2 = [(c[i] if i < len(c) else 0.0) - p2[i] for i in range(D)]
    #         n1 = NORM(v1); n2 = NORM(v2)
    #         cos = DOT(v1, v2) / (n1*n2 + eps)
    #         return (1.0 - cos) * (1.0 - cos)

    #     def _pospart(t):
    #         return 0.5*(t + sqrt(t*t + 1e-9))

    #     # ---- stage costs ----
    #     def l1_k(m, k_):
    #         p1 = POS(m.x1, k_, D); p2 = POS(m.x2, k_, D)
    #         u1 = [m.u1[k_, j] for j in m.U]
    #         stay_close = SUMSQ([p1[i] - p2[i] for i in range(D)])
    #         align = _align_penalty(p1, p2) if w1_align > 0.0 else 0.0
    #         return w1_close*stay_close + w1_align*align + c_eff1*SUMSQ(u1)

    #     def l2_k(m, k_):
    #         p1 = POS(m.x1, k_, D); p2 = POS(m.x2, k_, D)
    #         u2 = [m.u2[k_, j] for j in m.U]
    #         to_center = SUMSQ([p2[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
    #         d12 = NORM([p2[i] - p1[i] for i in range(D)])
    #         avoid = _pospart(d_safe - d12)**2
    #         return w2_c*to_center + w2_avoid*avoid + c_eff2*SUMSQ(u2)

    #     # ---- terminal costs ----
    #     def l1_T(m):
    #         p2T = POS(m.x2, T, D)
    #         to_center_T = SUMSQ([p2T[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
    #         return -w1_far_T * to_center_T

    #     def l2_T(m):
    #         p1T = POS(m.x1, T, D); p2T = POS(m.x2, T, D)
    #         to_center_T = SUMSQ([p2T[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
    #         d12T = NORM([p2T[i] - p1T[i] for i in range(D)])
    #         avoidT = _pospart(d_safe - d12T)**2
    #         return w2_c_T*to_center_T + w2_avoid_T*avoidT

    #     return l1_k, l2_k, l1_T, l2_T
    
    # elif k == "guard_center_v4":
    #     ar = (cfg.get("arena") or {})
    #     c = [float(ar.get(key, 0.0)) for key in (["cx","cy"] if D == 2 else ["cx","cy","cz"])]

    #     # --- attacker weights (as you had) ---
    #     w2_c       = float(cfg.get("w2_center",     1.0))
    #     w2_c_T     = float(cfg.get("w2_center_T",  20.0))
    #     c_eff2     = float(cfg.get("effort_w2",     1e-2))
    #     d_safe     = float(cfg.get("att_d_safe",    2.0))
    #     w2_avoid   = float(cfg.get("w2_avoid",      4.0))
    #     w2_avoid_T = float(cfg.get("w2_avoid_T",    1.0))

    #     # --- defender weights ---
    #     w1_close   = float(cfg.get("w1_close",      2.0))   # stay close to attacker (ram)
    #     c_eff1     = float(cfg.get("effort_w1",     1e-3))  # small so P1 is nimble
    #     w1_far_run = float(cfg.get("w1_far_run",    4.0))   # NEW: stage push of P2 away from center
    #     w1_far_T   = float(cfg.get("w1_far_T",     20.0))   # terminal push (you already had)

    #     eps        = float(cfg.get("eps_cost",      1e-9))

    #     def _pospart(t):
    #         return 0.5*(t + sqrt(t*t + 1e-9))

    #     def _to_center_sq(p):
    #         return SUMSQ([p[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])

    #     # ---- stage costs ----
    #     def l1_k(m, k_):
    #         p1 = POS(m.x1, k_, D); p2 = POS(m.x2, k_, D)
    #         u1 = [m.u1[k_, j] for j in m.U]

    #         # Rammer behavior: stay glued to P2, AND keep P2 far from center every step
    #         stay_close   = SUMSQ([p1[i] - p2[i] for i in range(D)])
    #         keep_p2_far  = - w1_far_run * _to_center_sq(p2)

    #         return w1_close*stay_close + keep_p2_far + c_eff1*SUMSQ(u1)

    #     def l2_k(m, k_):
    #         p1 = POS(m.x1, k_, D); p2 = POS(m.x2, k_, D)
    #         u2 = [m.u2[k_, j] for j in m.U]
    #         to_center = _to_center_sq(p2)
    #         d12       = NORM([p2[i] - p1[i] for i in range(D)])
    #         avoid     = _pospart(d_safe - d12)**2
    #         return w2_c*to_center + w2_avoid*avoid + c_eff2*SUMSQ(u2)

    #     # ---- terminal costs ----
    #     def l1_T(m):
    #         p2T = POS(m.x2, T, D)
    #         return - w1_far_T * _to_center_sq(p2T)

    #     def l2_T(m):
    #         p1T = POS(m.x1, T, D); p2T = POS(m.x2, T, D)
    #         tcT = _to_center_sq(p2T)
    #         d12T = NORM([p2T[i] - p1T[i] for i in range(D)])
    #         avoidT = _pospart(d_safe - d12T)**2
    #         return w2_c_T*tcT + w2_avoid_T*avoidT

    #     return l1_k, l2_k, l1_T, l2_T
    

    # elif k == "guard_center_v5":  # balloon-style, with terminal = sum of stages
    #     ar = (cfg.get("arena") or {})
    #     c  = [float(ar.get(k, 0.0)) for k in (["cx","cy"] if D==2 else ["cx","cy","cz"])]

    #     # ---- attacker params (mapped from Julia) ----
    #     P_pen_att   = float(cfg.get("P_penalty_att",    0.5))   # proximity_penalty
    #     P_pen_att_T = float(cfg.get("P_penalty_att_T",  0.5))   # kept for completeness
    #     w2_c        = float(cfg.get("w2_center",        1.0))   # weight on ||p2-c||^2
    #     w2_c_T      = float(cfg.get("w2_center_T",      0.0))   # not used since l_T=sum(l_k)
    #     r2          = float(cfg.get("R2",               1e-2))  # control_effort_weight

    #     # ---- defender params (mapped from Julia) ----
    #     P_pen_def   = float(cfg.get("P_penalty_def",   30.0))   # proximity_penalty_defender
    #     P_pen_def_T = float(cfg.get("P_penalty_def_T", 30.0))
    #     w1_far_T    = float(cfg.get("w1_far_T",        0.0))    # not used (terminal=sum stages)
    #     D_rate      = float(cfg.get("D_rate",          0.01))   # discouragement_rate
    #     Vmax_att    = float(cfg.get("Vmax_att",        1.0))    # set to attackerMax if available
    #     r1          = float(cfg.get("R1",              1e-4))   # defender_control_weight

    #     # ---- numerics (match Julia eps in denominators) ----
    #     eps = float(cfg.get("eps_cost", 1e-1))

    #     # local helpers
    #     def pos_(x, k_, D):   return [x[k_, i] for i in range(D)]
    #     def sumsq_(v):        return sum(vi*vi for vi in v)
    #     def norm_(v):         return sqrt(sumsq_(v) + eps)
    #     def to_center_sq_(p): return sumsq_([p[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])

    #     # ---------- stage costs ----------
    #     # Attacker: go to center + avoid defender + effort
    #     def l2_k(m, k_):
    #         p1 = pos_(m.x1, k_, D); p2 = pos_(m.x2, k_, D)
    #         u2 = [m.u2[k_, j] for j in m.U]
    #         dist_center   = w2_c * to_center_sq_(p2)
    #         prox_defender = P_pen_att / (eps + sumsq_([p2[i]-p1[i] for i in range(D)]))
    #         effort        = r2 * sumsq_(u2)
    #         return dist_center + prox_defender + effort

    #     # Defender: (aggressive) reward being close to attacker & keeping attacker far from center
    #     #           plus early-warning and speed-over-distance pressure, and small effort
    #     def l1_k(m, k_):
    #         p1 = pos_(m.x1, k_, D); p2 = pos_(m.x2, k_, D)
    #         u1 = [m.u1[k_, j] for j in m.U]
    #         d2c = norm_([p2[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
    #         dist_center_sq = d2c*d2c
    #         prox_term    = P_pen_def / (eps + sumsq_([p2[i]-p1[i] for i in range(D)]))
    #         warn_term    = exp(- d2c / D_rate)                    # big near center
    #         speed_over_d = 5.0 * Vmax_att / (eps + d2c)           # urgency if close to center
    #         effort       = r1 * sumsq_(u1)
    #         # maximize (prox_term + dist_center_sq + warn + speed/d) => minimize its negative
    #         return -(prox_term + dist_center_sq) + warn_term + speed_over_d + effort

    #     # ---------- terminal costs (non-zero): sum of stage costs ----------
    #     def l1_T(m):
    #         try:
    #             K = m.Ku     # 0..T-1
    #         except AttributeError:
    #             K = range(T)
    #         return sum(l1_k(m, k_) for k_ in K)

    #     def l2_T(m):
    #         try:
    #             K = m.Ku
    #         except AttributeError:
    #             K = range(T)
    #         return sum(l2_k(m, k_) for k_ in K)

    #     return l1_k, l2_k, l1_T, l2_T








    
    # ---- NEW: RL-weighted guard-center -----------------------------------------
    elif k == "guard_center_rl":
        # Center point (default origin or from arena)
        ar = (cfg.get("arena") or {})
        c = [float(ar.get(key, 0.0)) for key in (["cx","cy"] if D == 2 else ["cx","cy","cz"])]

        # Effort scales (kept as *features*, not hard-coded weights)
        eff_scale1 = float(cfg.get("eff_scale1", 1.0))
        eff_scale2 = float(cfg.get("eff_scale2", 1.0))

        # Optional safety epsilon for divisions
        eps = float(cfg.get("eps_cost", 1e-9))

        # -------- Feature maps (Pyomo expressions) --------
        # We expose small, well-behaved bases. You can add/remove easily.
        # Player 1 (defender) features φ1:
        #   φ1 = [ close_to_attacker, align_screen, effort1 ]
        def phi1(m, k_):
            p1 = _pos(m.x1, k_, D)
            p2 = _pos(m.x2, k_, D)
            u1 = [m.u1[k_, j] for j in m.U]

            close = _sumsq([p1[i] - p2[i] for i in range(D)])  # keep P1 close to P2
            # screening alignment: align (p1-p2) with (c-p2)
            v1 = [p1[i] - p2[i] for i in range(D)]
            v2 = [(c[i] if i < len(c) else 0.0) - p2[i] for i in range(D)]
            n1 = _norm(v1); n2 = _norm(v2)
            cos = _dot(v1, v2) / (n1*n2 + eps)
            align = (1.0 - cos)*(1.0 - cos)

            effort = eff_scale1 * _sumsq(u1)
            return [close, align, effort]

        # Player 2 (attacker) features φ2:
        #   φ2 = [ to_center, effort2 ]
        def phi2(m, k_):
            p2 = _pos(m.x2, k_, D)
            u2 = [m.u2[k_, j] for j in m.U]
            to_center = _sumsq([p2[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
            effort    = eff_scale2 * _sumsq(u2)
            return [to_center, effort]

        # ---- Terminal costs (fixed goals) ----
        def l1_T(m):
            # Defender wants attacker far from center -> minimize negative of distance^2
            p2T = _pos(m.x2, T, D)
            to_center_T = _sumsq([p2T[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
            return -to_center_T  # minimizing -> pushes attacker away

        def l2_T(m):
            # Attacker wants to be at center
            p2T = _pos(m.x2, T, D)
            to_center_T = _sumsq([p2T[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
            return to_center_T

        # ---- Weight providers (from RL) ----
        # You can pass callables that return a LIST/np.array of weights for the feature vector.
        # Signature: w*_provider(k: int) -> list[float]
        w1_provider = cfg.get("w1_provider", None)
        w2_provider = cfg.get("w2_provider", None)

        # Fallback to static defaults if not provided
        w1_default = list(cfg.get("w1_default", [2.0, 0.0, 0.01]))  # [close, align, effort1]
        w2_default = list(cfg.get("w2_default", [1.0, 0.01]))       # [to_center, effort2]

        # Optional normalization/bounding helper
        def _safe_weights(w_list, minv=0.0, maxv=10.0, l1norm=True):
            w = [float(max(min(x, maxv), minv)) for x in w_list]
            if l1norm:
                s = sum(w) + 1e-12
                w = [wi / s for wi in w]
            return w

        # ---- Stage costs combine weights with features ----
        def l1_k(m, k_):
            w = w1_provider(k_) if callable(w1_provider) else w1_default
            w = _safe_weights(w, minv=0.0, maxv=50.0, l1norm=False)  # let magnitudes matter if you want
            feats = phi1(m, k_)
            assert len(w) == len(feats), "w1 and phi1 length mismatch"
            return sum(w[i] * feats[i] for i in range(len(w)))

        def l2_k(m, k_):
            w = w2_provider(k_) if callable(w2_provider) else w2_default
            w = _safe_weights(w, minv=0.0, maxv=50.0, l1norm=False)
            feats = phi2(m, k_)
            assert len(w) == len(feats), "w2 and phi2 length mismatch"
            return sum(w[i] * feats[i] for i in range(len(w)))

        return l1_k, l2_k, l1_T, l2_T




    # ===================== lq (fallback / default) =====================
    else:
        q  = float(cfg.get("Q", 1.0))
        r1 = float(cfg.get("R1", 1.0))
        r2 = float(cfg.get("R2", 1.0))

        def l1_k(m, k_):
            return q*sum(m.x1[k_, i]**2 for i in m.S) + r1*sum(m.u1[k_, j]**2 for j in m.U)

        def l2_k(m, k_):
            return q*sum(m.x2[k_, i]**2 for i in m.S) + r2*sum(m.u2[k_, j]**2 for j in m.U)

        def l1_T(m):
            return q*sum(m.x1[T, i]**2 for i in m.S)

        def l2_T(m):
            return q*sum(m.x2[T, i]**2 for i in m.S)

        return l1_k, l2_k, l1_T, l2_T
