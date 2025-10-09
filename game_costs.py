# game_costs.py
# D-agnostic cost builder for the 2D/3D double-integrator game.
# Returns a list of per-player CasADi scalar objectives [f1, f2] (N=2).
from __future__ import annotations

import numpy as np
import casadi as ca

__all__ = ["build_costs"]


def build_costs(nx: int, nu: int, T: int, N: int, cfg: dict):
    """
    Build per-player objective functions for the N-player (N=2) game.
    The functions are D-agnostic for D in {2,3}.

    Parameters
    ----------
    nx, nu : int
        State and control dims (nx = 2D, nu = D).
    T : int
        Horizon length in states (there are T-1 control knots).
    N : int
        Number of players (currently assumes N=2 for unpacking).
    cfg : dict
        Scenario parameters. Key fields:
          - setting: {"baseline","boundary_hug","center_vs_block","orbit",
                      "rendezvous","antipodal","chase_escape_simple",
                      "fov_tag_simple","chase_escape_tail"}
          - arena: {"type": "box"|"circle"|"sphere", ...}
          - orbit_axis: 3-vector for 3D orbit setting
          - weights like effort_w1, effort_w2, etc.

    Returns
    -------
    list[callable]
        [f1, f2] where each f(tau, theta) returns a scalar MX.
    """
    setting = (cfg.get("setting", "baseline") or "baseline").lower()
    D = int(cfg.get("D", 3))               # trust the config
    assert D in (2, 3), "build_costs supports D in {2,3}"
    assert N == 2, "Current build_costs assumes N=2"
    nprim = T*nx + (T-1)*nu

    # ---------- Unpack both players from the stacked tau ----------
    def unpack_both(tau):
        tau1 = tau[0:nprim]
        tau2 = tau[nprim:2*nprim]
        def unpack_one(tau_p):
            X = tau_p[0:T*nx]; U = tau_p[T*nx:]
            xs = [X[i*nx:(i+1)*nx] for i in range(T)]
            us = [U[i*nu:(i+1)*nu] for i in range(T-1)]
            return xs, us
        return unpack_one(tau1), unpack_one(tau2)

    # ---------- Helpers (D-aware) ----------
    def pos(x): return x[0:D]
    def vel(x): return x[D:2*D]
    eps = 1e-6

    # 2D angular rate about origin (z-component)
    def omega_2d(p, v):
        x, y = p[0], p[1]
        vx, vy = v[0], v[1]
        r2 = x*x + y*y + 1e-3
        return (x*vy - y*vx) / r2

    # 3D scalar angular rate about a given axis (constant MX axis)
    # omega_axis = ax_hat · (p × v) / ||p||^2
    def omega_about_axis(p, v, ax_hat_mx):
        cx = p[1]*v[2] - p[2]*v[1]
        cy = p[2]*v[0] - p[0]*v[2]
        cz = p[0]*v[1] - p[1]*v[0]
        num = ax_hat_mx[0]*cx + ax_hat_mx[1]*cy + ax_hat_mx[2]*cz
        den = ca.sumsqr(p) + 1e-3
        return num / den

    # ---------- Axis & planar basis (NUMERIC, then cast to MX) ----------
    if D == 3:
        ax_np = np.array(cfg.get("orbit_axis", [0.0, 0.0, 1.0]), dtype=float)
        n_ax = np.linalg.norm(ax_np)
        if n_ax < 1e-12:
            ax_np = np.array([0.0, 0.0, 1.0], dtype=float)
            n_ax = 1.0
        ax_np = ax_np / n_ax
        ax_hat = ca.MX(ax_np)

        # choose a reference not parallel to axis
        ref = np.array([1.0, 0.0, 0.0]) if abs(ax_np[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u1_np = np.cross(ax_np, ref)
        n1 = np.linalg.norm(u1_np)
        if n1 < 1e-12:
            ref = np.array([0.0, 0.0, 1.0])
            u1_np = np.cross(ax_np, ref); n1 = np.linalg.norm(u1_np)
        u1_np = u1_np / (n1 + 1e-12)
        u2_np = np.cross(ax_np, u1_np)

        u1_axis = ca.MX(u1_np)
        u2_axis = ca.MX(u2_np)
    else:
        ax_hat  = None
        u1_axis = None
        u2_axis = None

    def azimuth_about_axis(p):
        if D == 2:
            return ca.atan2(p[1], p[0])
        # project onto plane orthogonal to axis: coords = [dot(p,u1), dot(p,u2)]
        pxp = ca.dot(p, u1_axis)
        pyp = ca.dot(p, u2_axis)
        return ca.atan2(pyp, pxp)

    # ---------- Defaults / params ----------
    R = None
    ar = cfg.get("arena", {})
    if ar.get("type", "box") in ("circle", "sphere"):
        R = float(ar["r"])
    if R is None: R = 3.0

    omega_target = float(cfg.get("omega_target", 0.5))
    effort_w1 = float(cfg.get("effort_w1", 0.01))
    effort_w2 = float(cfg.get("effort_w2", 0.01))
    plane_lock_w  = float(cfg.get("plane_lock_w", 0.0))   # keep close to orbit plane (3D)
    radial_lock_w = float(cfg.get("radial_lock_w", 0.0))  # discourage radial speed

    # center_vs_block
    w_center = float(cfg.get("w_center_def", 1.0))
    w_block  = float(cfg.get("w_block", 1.0))
    w_chase  = float(cfg.get("w_chase", 0.5))
    gamma    = float(cfg.get("block_gamma", 1.2))

    # Rendezvous target
    if "target" in cfg:
        target_vec = np.array(cfg["target"], dtype=float).reshape(-1)
        assert target_vec.size == D, "cfg['target'] must have length D"
        target = ca.MX(target_vec)
    else:
        target = ca.MX([R, 0.0]) if D == 2 else ca.MX([R, 0.0, 0.0])

    # ------------------ SCENARIOS (ROLE-SWAPPED) ------------------

    if setting == "chase_escape_tail":
        # P1 tails P2; P2 evades
        w_far   = float(cfg.get("w_far", 1.0))         # P2: push separation
        w_term  = float(cfg.get("w_term", 1.0))        # terminal weight (both)
        eff1    = float(cfg.get("effort_w1", 0.01))
        eff2    = float(cfg.get("effort_w2", 0.01))

        # Tail-follow parameters
        d_des   = float(cfg.get("follow_gap", 0.6))    # desired gap behind P2
        w_long  = float(cfg.get("w_tail_long", 1.0))   # weight on longitudinal error
        w_lat   = float(cfg.get("w_tail_lat", 8.0))    # weight on lateral error
        v_ref   = float(cfg.get("tail_v_ref", 0.2))    # speed scale for blend

        epsn = 1e-6

        def tail_cost(p_lead, p_fol, v_lead):
            # vhat ~ unit forward; well-behaved as speed->0
            speed = ca.sqrt(ca.sumsqr(v_lead) + epsn)
            vhat  = v_lead / (speed + epsn)

            r     = p_fol - p_lead
            spar  = ca.dot(r, vhat)              # >0 means follower is ahead of leader
            rlat  = r - spar * vhat              # lateral component

            # desired: spar ≈ -d_des (behind), and rlat ≈ 0 (aligned)
            tail_quad = w_long * (spar + d_des)**2 + w_lat * ca.sumsqr(rlat)

            # Blend with pure chase when leader is nearly stationary
            blend = (speed*speed) / (speed*speed + v_ref*v_ref)
            return blend * tail_quad + (1.0 - blend) * ca.sumsqr(r)

        def f1(tau, theta):
            # P1 tails P2
            (xs1, us1), (xs2, _) = unpack_both(tau)
            terms = []
            for t in range(T-1):
                p2, v2 = pos(xs2[t]), vel(xs2[t])  # leader
                p1     = pos(xs1[t])               # follower
                terms.append(tail_cost(p2, p1, v2) + eff1 * ca.sumsqr(us1[t]))
            # terminal tailing
            p2T, v2T = pos(xs2[T-1]), vel(xs2[T-1])
            p1T      = pos(xs1[T-1])
            terms.append(w_term * tail_cost(p2T, p1T, v2T))
            return ca.sum1(ca.vcat(terms))

        def f2(tau, theta):
            # P2 evades (maximize separation)
            (xs1, _), (xs2, us2) = unpack_both(tau)
            terms = []
            for t in range(T-1):
                r = pos(xs2[t]) - pos(xs1[t])
                terms.append(-w_far * ca.sumsqr(r) + eff2 * ca.sumsqr(us2[t]))
            rT = pos(xs2[T-1]) - pos(xs1[T-1])
            terms.append(-w_term * ca.sumsqr(rT))
            return ca.sum1(ca.vcat(terms))
        return [f1, f2]

    elif setting == "guard_center":
        # ---- params (tune in cfg) ----
        goal = np.asarray(cfg.get("goal", [0.0]*D), float).reshape(-1)
        goal_mx = ca.MX(goal)

        # proximity penalties
        P_def   = float(cfg.get("P_penalty_defender",  1.0))   # defender rewards being close to attacker (to block)
        P_att   = float(cfg.get("P_penalty_attacker",  1.0))   # attacker dislikes being close to defender
        eps_div = float(cfg.get("eps_div", 1e-2))               # ε for denominators (meters^2)

        # goal terms
        w_center   = float(cfg.get("w_center", 1.0))            # stage weight for ||x2 - goal||
        w_term     = float(cfg.get("w_term_center", 8.0))       # terminal weight for ||x2_T - goal||
        # optional ramp to emphasize later stages
        ramp_final = bool(cfg.get("ramp_final", True))

        # regularization
        w_u1 = float(cfg.get("effort_w1", 0.01))
        w_u2 = float(cfg.get("effort_w2", 0.01))
        w_v1 = float(cfg.get("w_v1", 0.0))                      # set >0 if you want velocity damping
        w_v2 = float(cfg.get("w_v2", 0.0))

        tiny = 1e-9

        def dist(a, b):
            # smooth Euclidean distance
            return ca.sqrt(ca.sumsqr(a - b) + tiny)

        def f2(tau, theta):
            # Attacker: go to goal, keep away from defender, pay effort
            (xs1, _), (xs2, us2) = unpack_both(tau)
            terms = []
            for t in range(T-1):
                p1, v1 = pos(xs1[t]), vel(xs1[t])
                p2, v2 = pos(xs2[t]), vel(xs2[t])
                w_t = (t+1)/T if ramp_final else 1.0

                d12_sq = ca.sumsqr(p2 - p1)
                go_goal = w_t * w_center * dist(p2, goal_mx)
                prox_repulse = P_att / (eps_div + d12_sq)       # 1/(ε+||x2-x1||^2)
                effort = w_u2 * ca.sumsqr(us2[t])
                vreg   = w_v2 * ca.sumsqr(v2)
                terms.append(go_goal + prox_repulse + effort + vreg)

            # terminal: strong at goal
            p2T, v2T = pos(xs2[T-1]), vel(xs2[T-1])
            terms.append(w_term * dist(p2T, goal_mx) + w_v2 * ca.sumsqr(v2T))
            return ca.sum1(ca.vcat(terms))

        def f1(tau, theta):
            # Defender: maximize (attacker near me) + (attacker far from goal), but we minimize,
            # so we negate those rewards.
            (xs1, us1), (xs2, _) = unpack_both(tau)
            terms = []
            for t in range(T-1):
                p1, v1 = pos(xs1[t]), vel(xs1[t])
                p2, v2 = pos(xs2[t]), vel(xs2[t])
                w_t = (t+1)/T if ramp_final else 1.0

                d12_sq = ca.sumsqr(p2 - p1)
                # Defender “reward” terms (we place a minus so minimization tries to enlarge them):
                reward_proximity =  P_def / (eps_div + d12_sq)  # wants attacker close
                reward_push_out  =  dist(p2, goal_mx)           # wants attacker far from goal
                # Convert to a minimization objective by negating:
                J_def_stage = -(reward_proximity + reward_push_out)

                effort = w_u1 * ca.sumsqr(us1[t])
                vreg   = w_v1 * ca.sumsqr(v1)
                terms.append(J_def_stage + effort + vreg)

            # terminal: same idea—prefer attacker far from goal (negate so minimization maximizes it)
            p2T, v1T = pos(xs2[T-1]), vel(xs1[T-1])
            terms.append(-w_term * dist(p2T, goal_mx) + w_v1 * ca.sumsqr(v1T))
            return ca.sum1(ca.vcat(terms))

        return [f1, f2]





    else:
        raise ValueError(f"Unknown setting '{setting}'.")
    
def add_roll_terms(J, tau_i, nx, nu, T, nx_tr, nu_tr, cfg):
    att  = cfg.get('att', {})
    if att.get('mode','') != 'roll1d':
        return J

    w_phi  = float(att.get('w_phi',  0.0))   # upright horizon
    w_dphi = float(att.get('w_dphi', 0.0))   # smoothness
    w_tau  = float(att.get('w_tau',  0.0))   # effort

    off_phi   = nx_tr
    off_u_tau = nu_tr

    # φ and Δφ
    for t in range(T):
        phi_t = tau_i[t*nx + off_phi]
        if w_phi > 0:   J += w_phi * phi_t**2
        if w_dphi > 0 and t < T-1:
            phi_n = tau_i[(t+1)*nx + off_phi]
            J += w_dphi * (phi_n - phi_t)**2

    # τx
    for t in range(T-1):
        tau_x = tau_i[T*nx + t*nu + off_u_tau]
        if w_tau > 0:   J += w_tau * tau_x**2

    return J