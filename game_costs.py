# game_costs.py
from __future__ import annotations
from typing import Tuple, Callable, Dict, Any
from pyomo.environ import sqrt

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
    k = (kind or "chase_escape_tail").lower()
    cfg = cfg or {}

    # ===================== chase_escape_tail =====================
    if k == "chase_escape_tail":
        c_eff1  = float(cfg.get("effort_w1",   0.01))
        c_eff2  = float(cfg.get("effort_w2",   0.01))
        c_wfar  = float(cfg.get("w_far",       1.0))
        c_wterm = float(cfg.get("w_term",      1.0))
        c_ddes  = float(cfg.get("follow_gap",  0.6))
        c_wlong = float(cfg.get("w_tail_long", 1.0))
        c_wlat  = float(cfg.get("w_tail_lat",  8.0))
        c_vref  = float(cfg.get("tail_v_ref",  0.2))
        eps_R   = float(cfg.get("path_eps_R",  1e-3))

        def tail_cost(pL, pF, vL):
            speed = _norm(vL)
            vhat = [vi / (speed + 1e-9) for vi in vL]
            r = [pF[i] - pL[i] for i in range(D)]
            s = _dot(r, vhat)               # parallel component
            rperp = [r[i] - s*vhat[i] for i in range(D)]
            tail = c_wlong*(s + c_ddes)**2 + c_wlat*_sumsq(rperp)
            blend = (speed*speed) / (speed*speed + c_vref*c_vref)
            return blend*tail + (1.0 - blend)*_sumsq(r)

        def l1_k(m, k_):
            p2 = _pos(m.x2, k_, D); v2 = _vel(m.x2, k_, D); p1 = _pos(m.x1, k_, D)
            u1 = [m.u1[k_, j] for j in m.U]
            return tail_cost(p2, p1, v2) + (c_eff1 + eps_R)*_sumsq(u1)

        def l2_k(m, k_):
            p1 = _pos(m.x1, k_, D); p2 = _pos(m.x2, k_, D)
            u2 = [m.u2[k_, j] for j in m.U]
            r  = [p2[i]-p1[i] for i in range(D)]
            return -c_wfar*_sumsq(r) + (c_eff2 + eps_R)*_sumsq(u2)

        def l1_T(m):
            p2 = _pos(m.x2, T, D); v2 = _vel(m.x2, T, D); p1 = _pos(m.x1, T, D)
            return c_wterm*tail_cost(p2, p1, v2)

        def l2_T(m):
            p1 = _pos(m.x1, T, D); p2 = _pos(m.x2, T, D)
            r  = [p2[i]-p1[i] for i in range(D)]
            return -c_wterm*_sumsq(r)

        return l1_k, l2_k, l1_T, l2_T

    # ===================== rendezvous_track =====================
    elif k == "rendezvous_track":
        # both agents want origin rendezvous; agent 2 weighs terminal closeness more
        w_run   = float(cfg.get("w_run", 1.0))
        w_term1 = float(cfg.get("w_term1", 5.0))
        w_term2 = float(cfg.get("w_term2", 10.0))
        r1      = float(cfg.get("R1", 1e-2))
        r2      = float(cfg.get("R2", 1e-2))

        def l1_k(m, k_):
            u1 = [m.u1[k_, j] for j in m.U]
            return w_run*sum(m.x1[k_, i]**2 for i in m.S) + r1*sum(ui**2 for ui in u1)

        def l2_k(m, k_):
            u2 = [m.u2[k_, j] for j in m.U]
            return w_run*sum(m.x2[k_, i]**2 for i in m.S) + r2*sum(ui**2 for ui in u2)

        def l1_T(m):
            return w_term1*sum(m.x1[T, i]**2 for i in m.S)

        def l2_T(m):
            return w_term2*sum(m.x2[T, i]**2 for i in m.S)

        return l1_k, l2_k, l1_T, l2_T
    
        # ===================== center_blocker =====================
    elif k == "center_blocker":
        # --- config ---
        # center point (list/tuple of length D); default origin
        c_list = cfg.get("center", [0.0]*D)
        c = [float(ci) for ci in c_list]

        # keep-out radius for the center
        R_keep  = float(cfg.get("R_keep", 3.0))

        # collision safety
        d_min   = float(cfg.get("d_min", 0.8))
        w_col   = float(cfg.get("w_col", 10.0))

        # attacker (P2) weights
        w_c_run = float(cfg.get("w_center_run",  1.0))
        w_c_T   = float(cfg.get("w_center_T",    8.0))
        r2      = float(cfg.get("R2",            1e-2))

        # blocker (P1) weights
        w_block     = float(cfg.get("w_block",       6.0))  # penalize P2 inside keep-out
        w_block_T   = float(cfg.get("w_block_T",    12.0))  # stronger at terminal
        w_hold_ctr  = float(cfg.get("w_hold_center", 0.2))  # keep P1 near center to block
        r1          = float(cfg.get("R1",            1e-2))

        eps = 1e-9
        def _pospart(t):
            # smooth positive part (C1): ~max(0,t)
            return 0.5*(t + sqrt(t*t + eps))

        def _p_minus_c(p):
            return [p[i] - c[i] for i in range(D)]

        def _collision_penalty(p1, p2):
            d12 = _norm([p1[i] - p2[i] for i in range(D)])
            return w_col * _pospart(d_min - d12)**2

        def _keepout_penalty(p2):
            dc = _norm(_p_minus_c(p2))
            return w_block * _pospart(R_keep - dc)**2

        def _keepout_penalty_T(p2):
            dc = _norm(_p_minus_c(p2))
            return w_block_T * _pospart(R_keep - dc)**2

        # --------- Player 1 (blocker) ---------
        # Stay near center to block + penalize if attacker enters keep-out + effort + collision
        def l1_k(m, k_):
            p1 = _pos(m.x1, k_, D)
            p2 = _pos(m.x2, k_, D)
            u1 = [m.u1[k_, j] for j in m.U]
            hold_center = w_hold_ctr * _sumsq(_p_minus_c(p1))
            keepout     = _keepout_penalty(p2)
            effort      = r1 * _sumsq(u1)
            collide     = _collision_penalty(p1, p2)
            return hold_center + keepout + effort + collide

        def l1_T(m):
            p1 = _pos(m.x1, T, D)
            p2 = _pos(m.x2, T, D)
            hold_center = w_hold_ctr * _sumsq(_p_minus_c(p1))
            keepoutT    = _keepout_penalty_T(p2)  # stronger terminal deterrence
            collide     = _collision_penalty(p1, p2)
            return hold_center + keepoutT + collide

        # --------- Player 2 (attacker) ---------
        # Go to center (stage+terminal) + effort + collision
        def l2_k(m, k_):
            p1 = _pos(m.x1, k_, D)
            p2 = _pos(m.x2, k_, D)
            u2 = [m.u2[k_, j] for j in m.U]
            to_center = w_c_run * _sumsq(_p_minus_c(p2))
            effort    = r2 * _sumsq(u2)
            collide   = _collision_penalty(p1, p2)
            return to_center + effort + collide

        def l2_T(m):
            p1 = _pos(m.x1, T, D)
            p2 = _pos(m.x2, T, D)
            to_centerT = w_c_T * _sumsq(_p_minus_c(p2))
            collide    = _collision_penalty(p1, p2)
            return to_centerT + collide

        return l1_k, l2_k, l1_T, l2_T
    
    # ===================== center_blocker_v2 =====================
    elif k == "center_blocker_v2":
        # center
        c_list = cfg.get("center", [0.0]*D)
        c = [float(ci) for ci in c_list]

        # ring radius where blocker prefers to sit (choose r_block > d_min and < R_keep)
        r_block   = float(cfg.get("r_block", 1.2))

        # (optional) keep-out region as a soft penalty; recommend enforcing via h_builders instead
        R_keep    = float(cfg.get("R_keep", 3.0))
        w_block   = float(cfg.get("w_block", 0.0))      # set 0.0 if you enforce it hard in h_builders
        w_block_T = float(cfg.get("w_block_T", 0.0))

        # P1 shaping weights
        w_ring     = float(cfg.get("w_ring", 10.0))     # hold P1 near ring radius r_block
        w_front    = float(cfg.get("w_front", 8.0))     # keep P1 radius <= P2 radius (stay "in front")
        w_align    = float(cfg.get("w_align", 3.0))     # align P1 on ray center->P2 (lateral offset)
        w_holdctr  = float(cfg.get("w_hold_center", 0.2))  # mild pre-positioning when P2 far
        r1         = float(cfg.get("R1", 1e-3))         # P1 effort (small so it moves)

        # P2 objective (attack center)
        w_c_run    = float(cfg.get("w_center_run", 1.0))
        w_c_T      = float(cfg.get("w_center_T", 8.0))
        r2         = float(cfg.get("R2", 3e-3))

        # numerics
        eps = 1e-9
        alpha = float(cfg.get("soft_norm_alpha", 0.05))  # >0 to avoid division blowups

        def _pospart(t):
            return 0.5*(t + sqrt(t*t + eps))  # smooth ReLU

        def _center_vec(p):
            return [p[i] - c[i] for i in range(D)]

        def _softnorm(v):
            # sqrt(||v||^2 + alpha^2) -- never below alpha, avoids 1/0
            return sqrt(sum(vi*vi for vi in v) + alpha*alpha)

        def _ring_penalty(p1):
            v = _center_vec(p1)
            n = _softnorm(v)
            # (||p1-c|| - r_block)^2
            return (n - r_block)**2

        def _front_penalty(p1, p2):
            # penalize if ||p1-c|| > ||p2-c||   (i.e., P1 is "outside" attacker, not blocking)
            n1 = _softnorm(_center_vec(p1))
            n2 = _softnorm(_center_vec(p2))
            return _pospart(n1 - n2)**2

        def _align_penalty(p1, p2):
            # squared lateral offset of p1 from ray center->p2:
            # ||(I - u u^T)(p1-c)||^2 with u = (p2-c)/softnorm(p2-c)
            vp2 = _center_vec(p2)
            n = _softnorm(vp2)
            u = [vp2[i]/n for i in range(D)]                 # soft unit
            v1 = _center_vec(p1)
            # proj length onto u:
            dot = sum(u[i]*v1[i] for i in range(D))
            # lateral component squared = ||v1||^2 - (u·v1)^2
            return sum(vi*vi for vi in v1) - dot*dot

        def _keepout_stage(p2):
            if w_block <= 0.0:
                return 0.0
            n = _softnorm(_center_vec(p2))
            return w_block * _pospart(R_keep - n)**2

        def _keepout_term(p2):
            if w_block_T <= 0.0:
                return 0.0
            n = _softnorm(_center_vec(p2))
            return w_block_T * _pospart(R_keep - n)**2

        # --------- Player 1 (blocker) ---------
        def l1_k(m, k_):
            p1 = _pos(m.x1, k_, D)
            p2 = _pos(m.x2, k_, D)
            u1 = [m.u1[k_, j] for j in m.U]
            ring   = w_ring * _ring_penalty(p1)
            front  = w_front * _front_penalty(p1, p2)
            align  = w_align * _align_penalty(p1, p2)
            hold   = w_holdctr * sum((p1[i]-c[i])**2 for i in range(D))
            keep   = _keepout_stage(p2)   # usually 0 when enforcing via h_builders
            effort = r1 * sum(ui*ui for ui in u1)
            return ring + front + align + hold + keep + effort

        def l1_T(m):
            p1 = _pos(m.x1, T, D)
            p2 = _pos(m.x2, T, D)
            ring   = w_ring * _ring_penalty(p1)
            front  = w_front * _front_penalty(p1, p2)
            align  = w_align * _align_penalty(p1, p2)
            hold   = w_holdctr * sum((p1[i]-c[i])**2 for i in range(D))
            keepT  = _keepout_term(p2)
            return ring + front + align + hold + keepT

        # --------- Player 2 (attacker) ---------
        def l2_k(m, k_):
            p2 = _pos(m.x2, k_, D)
            u2 = [m.u2[k_, j] for j in m.U]
            to_center = w_c_run * sum((p2[i]-c[i])**2 for i in range(D))
            effort    = r2 * sum(ui*ui for ui in u2)
            return to_center + effort

        def l2_T(m):
            p2 = _pos(m.x2, T, D)
            to_centerT = w_c_T * sum((p2[i]-c[i])**2 for i in range(D))
            return to_centerT

        return l1_k, l2_k, l1_T, l2_T
    
    elif k == "center_rammer":
        # center & radii
        c_list = cfg.get("center", [0.0]*D)
        c = [float(ci) for ci in c_list]

        r_block   = float(cfg.get("r_block", 1.2))    # desired P1 ring radius (choose > d_min)
        R_keep    = float(cfg.get("R_keep", 3.0))     # optional soft keep-out; prefer hard via h_builders

        # weights
        w_ring    = float(cfg.get("w_ring", 8.0))     # keep P1 near r_block
        w_front   = float(cfg.get("w_front", 6.0))    # keep P1 radius <= P2 radius
        w_align   = float(cfg.get("w_align", 2.0))    # reduce lateral offset of P1 from center->P2 ray
        w_push    = float(cfg.get("w_push", 20.0))    # NEW: enforce outward radial separation (ramming)
        w_holdctr = float(cfg.get("w_hold_center", 0.2))
        w_block   = float(cfg.get("w_block", 0.0))    # soft center keep-out for P2 (stage)
        w_block_T = float(cfg.get("w_block_T", 0.0))  # soft center keep-out for P2 (terminal)

        # efforts
        r1 = float(cfg.get("R1", 8e-4))               # make P1 nimble
        r2 = float(cfg.get("R2", 3e-3))

        # attacker drive to center
        w_c_run = float(cfg.get("w_center_run", 0.7)) # slightly softened so P1 can influence
        w_c_T   = float(cfg.get("w_center_T", 6.0))

        # numerics
        eps   = 1e-9
        alpha = float(cfg.get("soft_norm_alpha", 0.05))  # softnorm floor
        d_min = float(cfg.get("d_min", 0.8))             # must match separation constraint

        def _pospart(t):
            return 0.5*(t + sqrt(t*t + eps))  # smooth hinge

        def _cv(p):  # center->p
            return [p[i] - c[i] for i in range(D)]

        def _softnorm(v):
            return sqrt(sum(vi*vi for vi in v) + alpha*alpha)

        def _ring_penalty(p1):
            n = _softnorm(_cv(p1))
            return (n - r_block)**2

        def _front_penalty(p1, p2):
            # penalize if P1 is farther from center than P2 (not blocking)
            n1 = _softnorm(_cv(p1))
            n2 = _softnorm(_cv(p2))
            return _pospart(n1 - n2)**2

        def _align_penalty(p1, p2):
            # lateral offset of P1 from the ray center->P2
            vp2 = _cv(p2); n = _softnorm(vp2); u = [vp2[i]/n for i in range(D)]
            v1  = _cv(p1)
            dot = sum(u[i]*v1[i] for i in range(D))
            return sum(vi*vi for vi in v1) - dot*dot

        def _radial_push_hinge(p1, p2):
            # encourage (p2 - p1) to have outward radial component >= d_min
            vp2 = _cv(p2); n = _softnorm(vp2); u_r = [vp2[i]/n for i in range(D)]     # soft unit radial at p2
            sep = [p2[i] - p1[i] for i in range(D)]
            proj = sum(u_r[i]*sep[i] for i in range(D))                               # outward component
            return _pospart(d_min - proj)**2

        def _keepout_stage(p2):
            if w_block <= 0.0: return 0.0
            n = _softnorm(_cv(p2))
            return w_block * _pospart(R_keep - n)**2

        def _keepout_term(p2):
            if w_block_T <= 0.0: return 0.0
            n = _softnorm(_cv(p2))
            return w_block_T * _pospart(R_keep - n)**2

        # --------- Player 1 (blocker) ---------
        def l1_k(m, k_):
            p1 = _pos(m.x1, k_, D)
            p2 = _pos(m.x2, k_, D)
            u1 = [m.u1[k_, j] for j in m.U]
            ring   = w_ring * _ring_penalty(p1)
            front  = w_front * _front_penalty(p1, p2)
            align  = w_align * _align_penalty(p1, p2)
            push   = w_push * _radial_push_hinge(p1, p2)     # <-- the “ramming” driver
            hold   = w_holdctr * sum((p1[i]-c[i])**2 for i in range(D))
            keep   = _keepout_stage(p2)
            effort = r1 * sum(ui*ui for ui in u1)
            return ring + front + align + push + hold + keep + effort

        def l1_T(m):
            p1 = _pos(m.x1, T, D)
            p2 = _pos(m.x2, T, D)
            ring   = w_ring * _ring_penalty(p1)
            front  = w_front * _front_penalty(p1, p2)
            align  = w_align * _align_penalty(p1, p2)
            push   = w_push * _radial_push_hinge(p1, p2)
            hold   = w_holdctr * sum((p1[i]-c[i])**2 for i in range(D))
            keepT  = _keepout_term(p2)
            return ring + front + align + push + hold + keepT

        # --------- Player 2 (attacker) ---------
        def l2_k(m, k_):
            p2 = _pos(m.x2, k_, D)
            u2 = [m.u2[k_, j] for j in m.U]
            to_center = w_c_run * sum((p2[i]-c[i])**2 for i in range(D))
            effort    = r2 * sum(ui*ui for ui in u2)
            return to_center + effort

        def l2_T(m):
            p2 = _pos(m.x2, T, D)
            to_centerT = w_c_T * sum((p2[i]-c[i])**2 for i in range(D))
            return to_centerT

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
    

def build_game_costs_N(
    kind: str,
    cfg: Dict[str, Any],
    D: int,
    T: int,
    N: int,
) -> Tuple[List[Callable], List[Callable]]:
    """
    Returns:
      l_k: list (len N) of callables l_k[p-1](m,k)   for k=0..T-1
      l_T: list (len N) of callables l_T[p-1](m)     terminal at k=T

    IMPORTANT: This version assumes explicit per-player components on the model:
      m.x1[k,i], m.x2[k,i], ..., m.xN[k,i]
      m.u1[k,j], m.u2[k,j], ..., m.uN[k,j]
    """

    kname = (kind or "generic_n").lower()
    cfg = cfg or {}

    # ----- helpers that respect explicit per-player components -----
    def _Xp(m, p): return getattr(m, f"x{p}")
    def _Up(m, p): return getattr(m, f"u{p}")

    def _pos(m, p, k): return [_Xp(m,p)[k, i] for i in range(D)]
    def _vel(m, p, k): return [_Xp(m,p)[k, i] for i in range(D, 2*D)]
    def _u(m, p, k):   return [_Up(m,p)[k, j] for j in m.U]

    def _sq(v):        return sum(vi*vi for vi in v)
    def _norm(v):      return sqrt(_sq(v) + 1e-9)
    def _dot(a, b):    return sum(a[i]*b[i] for i in range(len(a)))

    # ---------------------- generic_n (LQ-ish) ----------------------
    if kname == "generic_n":
        q  = float(cfg.get("Q", 1.0))
        r  = float(cfg.get("R", 1e-2))
        l_k, l_T = [], []
        for p in range(1, N+1):
            def _lk(m, k, p=p):
                # use full nx in m.S (safe even if nx != 2D)
                return q*sum(_Xp(m,p)[k, i]**2 for i in m.S) + r*sum(_Up(m,p)[k, j]**2 for j in m.U)
            def _lT(m, p=p):
                return q*sum(_Xp(m,p)[T, i]**2 for i in m.S)
            l_k.append(_lk); l_T.append(_lT)
        return l_k, l_T

    # ---------------------- chase_escape_tail (exact 2p behavior only) ----------------------
    if kname in ("chase_escape_tail", "chase_escape_tail_n") and N == 2:
        # Same knobs as legacy 2p
        c_eff1  = float(cfg.get("effort_w1",   0.01))
        c_eff2  = float(cfg.get("effort_w2",   0.01))
        c_wfar  = float(cfg.get("w_far",       1.0))
        c_wterm = float(cfg.get("w_term",      1.0))
        c_ddes  = float(cfg.get("follow_gap",  0.6))
        c_wlong = float(cfg.get("w_tail_long", 1.0))
        c_wlat  = float(cfg.get("w_tail_lat",  8.0))
        c_vref  = float(cfg.get("tail_v_ref",  0.2))
        eps_R   = float(cfg.get("path_eps_R",  1e-3))

        def _sumsq(a):  return sum(ai*ai for ai in a)

        # follower wants to sit at (s=-d_des, r_perp=0) behind leader
        def _tail_cost(pL, pF, vL):
            speed = _norm(vL)
            vhat = [vi / (speed + 1e-9) for vi in vL]
            r    = [pF[i] - pL[i] for i in range(D)]
            s    = _dot(r, vhat)
            rperp = [r[i] - s*vhat[i] for i in range(D)]
            tail  = c_wlong*(s + c_ddes)**2 + c_wlat*_sumsq(rperp)
            blend = (speed*speed) / (speed*speed + c_vref*c_vref)
            return blend*tail + (1.0 - blend)*_sumsq(r)

        # Player 1 = follower, Player 2 = leader
        def _l1_k(m, k):
            p2 = _pos(m, 2, k)
            v2 = _vel(m, 2, k)
            p1 = _pos(m, 1, k)
            u1 = _u(m, 1, k)
            return _tail_cost(p2, p1, v2) + (c_eff1 + eps_R)*_sq(u1)

        def _l2_k(m, k):
            p1 = _pos(m, 1, k)
            p2 = _pos(m, 2, k)
            u2 = _u(m, 2, k)
            r  = [p2[i] - p1[i] for i in range(D)]
            return -c_wfar*_sq(r) + (c_eff2 + eps_R)*_sq(u2)

        def _l1_T(m):
            p2 = [_Xp(m,2)[T, i] for i in range(D)]
            v2 = [_Xp(m,2)[T, i] for i in range(D, 2*D)]
            p1 = [_Xp(m,1)[T, i] for i in range(D)]
            return c_wterm * _tail_cost(p2, p1, v2)

        def _l2_T(m):
            p1 = [_Xp(m,1)[T, i] for i in range(D)]
            p2 = [_Xp(m,2)[T, i] for i in range(D)]
            r  = [p2[i] - p1[i] for i in range(D)]
            return -c_wterm * _sq(r)

        return [_l1_k, _l2_k], [_l1_T, _l2_T]

    # ---------------------- 3p_center_blocker ----------------------
    if kname == "3p_center_blocker":
        c = [float(ci) for ci in cfg.get("center", [0.0]*D)]
        R_keep  = float(cfg.get("R_keep", 3.0))
        w_block = float(cfg.get("w_block", 8.0))
        w_T     = float(cfg.get("w_T", 12.0))
        r1      = float(cfg.get("R1", 1e-3))
        rA      = float(cfg.get("R_att", 3e-3))
        eps=1e-9
        def _relu(t): return 0.5*(t + sqrt(t*t + eps))

        def _d_center(m, p, k):
            return _norm([_pos(m,p,k)[i]-c[i] for i in range(D)])

        # attacker stage/terminal
        def _att_lk(p):
            return lambda m,k, p=p: _d_center(m,p,k)**2 + rA*_sq(_u(m,p,k))

        def _att_lT(p):
            return lambda m, p=p: _norm([_Xp(m,p)[T,i] - c[i] for i in range(D)])**2 * w_T

        # blocker penalizes attackers inside keep-out
        def _blk_lk():
            def f(m,k):
                pen = 0.0
                for pa in range(2, N+1):
                    d = _d_center(m, pa, k)
                    pen += w_block * _relu(R_keep - d)**2
                return pen + r1*_sq(_u(m,1,k))
            return f

        def _blk_lT():
            def f(m):
                pen = 0.0
                for pa in range(2, N+1):
                    d = _norm([_Xp(m,pa)[T,i]-c[i] for i in range(D)])
                    pen += w_T * _relu(R_keep - d)**2
                return pen
            return f

        l_k=[None]*N; l_T=[None]*N
        l_k[0] = _blk_lk(); l_T[0] = _blk_lT()
        for p in range(2, N+1):
            l_k[p-1] = _att_lk(p); l_T[p-1] = _att_lT(p)
        return l_k, l_T

    # ---------------------- 4p_tag (one runner vs 3 chasers) ----------------------
    if kname == "4p_tag":
        runner = int(cfg.get("runner", 1))  # 1..N
        w_sep  = float(cfg.get("w_sep", 4.0))
        rU     = float(cfg.get("R_u", 1e-2))

        def _pair_sq(m,k,pa,pb):
            a = _pos(m,pa,k); b = _pos(m,pb,k)
            return sum((a[i]-b[i])**2 for i in range(D))

        l_k=[None]*N; l_T=[None]*N
        for p in range(1, N+1):
            if p == runner:
                # runner maximizes separation (we minimize negative)
                def _lk(m,k,p=p):
                    sep = 0.0
                    for q in range(1, N+1):
                        if q == p: continue
                        sep += _pair_sq(m,k,p,q)
                    return -w_sep*sep + rU*_sq(_u(m,p,k))
                def _lT(m,p=p):
                    sepT=0.0
                    a = [_Xp(m,p)[T,i] for i in range(D)]
                    for q in range(1,N+1):
                        if q == p: continue
                        b = [_Xp(m,q)[T,i] for i in range(D)]
                        sepT += sum((a[i]-b[i])**2 for i in range(D))
                    return -w_sep*sepT
            else:
                # chaser minimizes distance to runner
                def _lk(m,k,p=p):
                    return _pair_sq(m,k,p,runner) + rU*_sq(_u(m,p,k))
                def _lT(m,p=p):
                    a=[_Xp(m,p)[T,i] for i in range(D)]
                    b=[_Xp(m,runner)[T,i] for i in range(D)]
                    return sum((a[i]-b[i])**2 for i in range(D))
            l_k[p-1]=_lk; l_T[p-1]=_lT
        return l_k, l_T

    # ---------------------- 5p_teams_3v2 (team sums) ----------------------
    if kname == "5p_teams_3v2":
        teamA = list(cfg.get("teamA", [1,2,3]))
        teamB = [p for p in range(1,N+1) if p not in teamA]
        w_c = float(cfg.get("w_cohesion", 0.2))
        w_h = float(cfg.get("w_hunt", 1.0))
        rU  = float(cfg.get("R_u", 1e-2))

        def _centroid(m,k,team):
            return [ sum(_Xp(m,p)[k,i] for p in team)/len(team) for i in range(D) ]
        def _sq_to_point(v, w):
            return sum((v[i]-w[i])**2 for i in range(D))

        l_k=[None]*N; l_T=[None]*N
        for p in range(1,N+1):
            if p in teamA:
                def _lk(m,k,p=p):
                    cA = _centroid(m,k,teamA); cB = _centroid(m,k,teamB)
                    return w_h*_sq_to_point(_pos(m,p,k), cB) + w_c*_sq_to_point(_pos(m,p,k), cA) + rU*_sq(_u(m,p,k))
                def _lT(m,p=p):
                    cB = [sum(_Xp(m,q)[T,i] for q in teamB)/len(teamB) for i in range(D)]
                    a  = [_Xp(m,p)[T,i] for i in range(D)]
                    return w_h*sum((a[i]-cB[i])**2 for i in range(D))
            else:
                def _lk(m,k,p=p):
                    cB = _centroid(m,k,teamB); cA = _centroid(m,k,teamA)
                    return w_h*_sq_to_point(_pos(m,p,k), cA) + w_c*_sq_to_point(_pos(m,p,k), cB) + rU*_sq(_u(m,p,k))
                def _lT(m,p=p):
                    cA = [sum(_Xp(m,q)[T,i] for q in teamA)/len(teamA) for i in range(D)]
                    a  = [_Xp(m,p)[T,i] for i in range(D)]
                    return w_h*sum((a[i]-cA[i])**2 for i in range(D))
            l_k[p-1]=_lk; l_T[p-1]=_lT
        return l_k, l_T

    # fallback
    return build_game_costs_N("generic_n", cfg, D, T, N)