# game_costs.py
# D-agnostic cost builder for the 2D/3D double-integrator game.
# Returns a list of per-player CasADi scalar objectives [f1, f2] (N=2).
from __future__ import annotations

import numpy as np
import casadi as ca

__all__ = ["build_costs_pyomo"]

def build_costs_pyomo(nx, nu, T, N, cfg):
    """
    Return [f1_pyomo, f2_pyomo] where each f_i(tau_all, theta) builds a Pyomo scalar expr.
    tau_all = concatenation of player 1 and 2 primals (Pyomo VarData lists).
    """
    D = int(cfg.get("D", 3))
    nprim = T*nx + (T-1)*nu

    # weights / knobs (lift from your cfg)
    eff1 = float(cfg.get("effort_w1", 1.0))
    eff2 = float(cfg.get("effort_w2", 1.0))
    epsn = 1e-9

    # stage weights, if any
    def w_stage(t):  # same for all t by default
        return 1.0

    def _split_players(tau_all):
        tau1 = tau_all[0:nprim]
        tau2 = tau_all[nprim:2*nprim]
        xs1, us1 = py_unpack_tau(tau1, nx, nu, T)
        xs2, us2 = py_unpack_tau(tau2, nx, nu, T)
        return xs1, us1, xs2, us2

    def pos(x): return x[0:D]
    def vel(x): return x[D:2*D]

    # Tail cost: penalize follower (1) getting behind leader (2)
    # (Adjust if your original “chase_escape_tail” definition was flipped)
    def tail_cost(p_lead, p_fol, v_lead):
        # vhat = v_lead / ||v_lead||
        speed = py_norm2(v_lead, eps=epsn)
        vhat  = [vj / (speed + epsn) for vj in v_lead]
        r     = [pf - pl for pf,pl in zip(p_fol, p_lead)]
        spar  = py_dot(r, vhat)  # projection along v_lead direction
        # Encourage small spar (follower near “tail”)
        return spar*spar

    # --- f1: follower 1 tries to tail 2 + effort on u1
    def f1(tau_all, theta):
        xs1, us1, xs2, us2 = _split_players(tau_all)
        terms = []
        for t in range(T-1):
            p2, v2 = pos(xs2[t]), vel(xs2[t])
            p1     = pos(xs1[t])
            stage  = tail_cost(p_lead=p2, p_fol=p1, v_lead=v2) + eff1 * py_sumsqr(us1[t])
            terms.append(w_stage(t) * stage)
        # terminal (optional): keep consistent with your CasADi code if it had one
        p2T, v2T = pos(xs2[T-1]), vel(xs2[T-1])
        p1T      = pos(xs1[T-1])
        terms.append(tail_cost(p_lead=p2T, p_fol=p1T, v_lead=v2T))
        return sum(terms)

    # --- f2: leader 2 tries to escape 1 + effort on u2
    def f2(tau_all, theta):
        xs1, us1, xs2, us2 = _split_players(tau_all)
        terms = []
        for t in range(T-1):
            p1, v1 = pos(xs1[t]), vel(xs1[t])
            p2     = pos(xs2[t])
            # negative of tail: leader wants large spar (push follower behind)
            # i.e., minimize -spar^2 = maximize spar^2; use minus sign:
            stage  = (-1.0)*tail_cost(p_lead=p2, p_fol=p1, v_lead=v2) + eff2 * py_sumsqr(us2[t])
            terms.append(w_stage(t) * stage)
        p1T, v1T = pos(xs1[T-1]), vel(xs1[T-1])
        p2T      = pos(xs2[T-1])
        terms.append((-1.0)*tail_cost(p_lead=p2T, p_fol=p1T, v_lead=v1T))
        return sum(terms)

    return [f1, f2]
