#3Player_utils.py



# ---------- explicit 3-player costs ----------
def build_game_costs_3p(kind: str, cfg: dict, D: int, T: int):
    """
    Returns:
      l_k: [l1_k, l2_k, l3_k] where l*_k(m,k) is stage cost at time k
      l_T: [l1_T, l2_T, l3_T] where l*_T(m)   is terminal cost
    """
    kind = (kind or "reach_boundary_3p").lower()
    print(kind)
    cfg  = dict(cfg or {})

    def _Xp(m, p): return getattr(m, f"x{p}")
    def _Up(m, p): return getattr(m, f"u{p}")
    def _pos(m, p, k): return [_Xp(m,p)[k, i] for i in range(D)]
    def _u(m, p, k):   return [_Up(m,p)[k, j] for j in m.U]
    def _sq(v):        return sum(vi*vi for vi in v)
    def _norm(v):      return (_sq(v) + 1e-9)**0.5

    # ---- all players reach the boundary of a given sphere (centered at c, radius R) ----
    if kind in ("reach_boundary_3p", "reach_boundary"):
        c = [float(cfg.get("arena", {}).get(k, 0.0)) for k in (["cx","cy"] if D==2 else ["cx","cy","cz"])]
        R = float(cfg.get("arena", {}).get("r", cfg.get("R", 20.0)))
        Qr   = float(cfg.get("Q_radial",   1.0))
        Qr_T = float(cfg.get("Q_radial_T", 20.0))
        Ru   = float(cfg.get("R_u",        1e-3))

        def _radial_err_sq(m, p, k):
            pvec = _pos(m, p, k)
            d    = _norm([pvec[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
            return (R - d)**2

        l_k = []
        l_T = []
        for p in (1,2,3):
            def _lk(m,k,p=p):
                return Qr*_radial_err_sq(m, p, k) + Ru*_sq(_u(m, p, k))
            def _lT(m,p=p):
                pT = [_Xp(m,p)[T, i] for i in range(D)]
                dT = _norm([pT[i] - (c[i] if i < len(c) else 0.0) for i in range(D)])
                return Qr_T*(R - dT)**2
            l_k.append(_lk); l_T.append(_lT)
        return l_k, l_T

    # ---- simple LQ fallback (energy in x,u) ----
    if kind in ("lq", "generic"):
        q = float(cfg.get("Q", 1.0))
        r = float(cfg.get("R", 1e-2))
        l_k, l_T = [], []
        for p in (1,2,3):
            def _lk(m,k,p=p):
                return q*sum(getattr(m,f"x{p}")[k,i]**2 for i in m.S) + r*sum(getattr(m,f"u{p}")[k,j]**2 for j in m.U)
            def _lT(m,p=p):
                return q*sum(getattr(m,f"x{p}")[T,i]**2 for i in m.S)
            l_k.append(_lk); l_T.append(_lT)
        return l_k, l_T

    # default
    return ValueError("Set a valid something")


# ---------- explicit 3-player MCP ----------
from pyomo.environ import ConcreteModel, Var, Param, RangeSet, Reals, NonNegativeReals, Constraint, value
from pyomo.environ import differentiate
from pyomo.core.expr.calculus.derivatives import Modes
from pyomo.mpec import Complementarity, complements
import numpy as np

def build_mcp_three_player_one_shot(
    Ad, Bd, T, nx, nu,
    x0_1, x0_2, x0_3,
    D=3,
    x_var_box=(-1e6, 1e6),
    u_var_box=(-1e3, 1e3),
    h_builders=None,                     # list of callables h(m,k) >= 0
    cost_kind="reach_boundary_3p",
    cost_cfg=None,
):
    """
    One MCP for a 3-player horizon game.

    Primals: τ = {x1(k),u1(k), x2(k),u2(k), x3(k),u3(k)} for k=0..T
    Shared eqs: g̃ = [IC1,dyn1, IC2,dyn2, IC3,dyn3] with multipliers λ̃
    Shared ineq: h̃ >= 0 with μ̃ ≥ 0, μ̃ ⟂ h̃
    Stationarity: ∇f_p - Jg̃^T λ̃ - Jh̃^T μ̃ = 0 on each player's variables
    """
    cost_cfg = cost_cfg or {}
    m = ConcreteModel()

    # ---------- sets ----------
    m.Kx = RangeSet(0, T)
    m.Ku = RangeSet(0, T-1)
    m.S  = RangeSet(0, nx-1)
    m.U  = RangeSet(0, nu-1)

    # ---------- params ----------
    Ad = np.asarray(Ad, float); Bd = np.asarray(Bd, float)
    x0_1 = np.asarray(x0_1, float); x0_2 = np.asarray(x0_2, float); x0_3 = np.asarray(x0_3, float)

    m.A = Param(m.S, m.S, initialize=lambda _m,i,j: float(Ad[i,j]), within=Reals, mutable=False)
    m.B = Param(m.S, m.U, initialize=lambda _m,i,j: float(Bd[i,j]), within=Reals, mutable=False)

    # Make IC params MUTABLE so RHC can update each turn
    m.x01 = Param(m.S, initialize=lambda _m,i: float(x0_1[i]), within=Reals, mutable=True)
    m.x02 = Param(m.S, initialize=lambda _m,i: float(x0_2[i]), within=Reals, mutable=True)
    m.x03 = Param(m.S, initialize=lambda _m,i: float(x0_3[i]), within=Reals, mutable=True)

    # ---------- variables ----------
    xlb, xub = float(x_var_box[0]), float(x_var_box[1])
    ulb, uub = float(u_var_box[0]), float(u_var_box[1])

    m.x1 = Var(m.Kx, m.S, bounds=lambda _m,k,i: (xlb, xub))
    m.x2 = Var(m.Kx, m.S, bounds=lambda _m,k,i: (xlb, xub))
    m.x3 = Var(m.Kx, m.S, bounds=lambda _m,k,i: (xlb, xub))
    m.u1 = Var(m.Ku, m.U, bounds=lambda _m,k,j: (ulb, uub))
    m.u2 = Var(m.Ku, m.U, bounds=lambda _m,k,j: (ulb, uub))
    m.u3 = Var(m.Ku, m.U, bounds=lambda _m,k,j: (ulb, uub))

    # ---------- shared equality constraints ----------
    def g_ic_expr(_m, p, i):
        return getattr(_m, f"x{p}")[0, i] - getattr(_m, f"x0{p}")[i]
    def g_dyn_expr(_m, p, k, i):
        x = getattr(_m, f"x{p}"); u = getattr(_m, f"u{p}")
        return x[k+1, i] - sum(_m.A[i,j]*x[k,j] for j in _m.S) - sum(_m.B[i,j]*u[k,j] for j in _m.U)

    # ICs
    m.ic1 = Constraint(m.S, rule=lambda _m,i: g_ic_expr(_m, 1, i) == 0)
    m.ic2 = Constraint(m.S, rule=lambda _m,i: g_ic_expr(_m, 2, i) == 0)
    m.ic3 = Constraint(m.S, rule=lambda _m,i: g_ic_expr(_m, 3, i) == 0)
    # Dynamics
    m.dyn1 = Constraint(m.Ku, m.S, rule=lambda _m,k,i: g_dyn_expr(_m, 1, k, i) == 0)
    m.dyn2 = Constraint(m.Ku, m.S, rule=lambda _m,k,i: g_dyn_expr(_m, 2, k, i) == 0)
    m.dyn3 = Constraint(m.Ku, m.S, rule=lambda _m,k,i: g_dyn_expr(_m, 3, k, i) == 0)

    # ---------- multipliers for equalities (free) ----------
    m.lam_ic1 = Var(m.S, domain=Reals); m.lam_ic2 = Var(m.S, domain=Reals); m.lam_ic3 = Var(m.S, domain=Reals)
    m.lam_dyn1 = Var(m.Ku, m.S, domain=Reals); m.lam_dyn2 = Var(m.Ku, m.S, domain=Reals); m.lam_dyn3 = Var(m.Ku, m.S, domain=Reals)

    # ---------- shared inequalities h >= 0 with μ ≥ 0 and μ ⟂ h ----------
    h_builders = list(h_builders or [])
    if h_builders:
        m.H = RangeSet(0, len(h_builders)-1)
        m.mu = Var(m.H, m.Kx, domain=NonNegativeReals)
        def _h_expr(_m, h, k):
            return h_builders[int(h)](_m, k)
        m.h_comp = Complementarity(
            m.H, m.Kx,
            rule=lambda _m,h,k: complements(_m.mu[h,k] >= 0, _h_expr(_m,h,k) >= 0)
        )

    # ---------- costs ----------

    #raise("Debug")
    l_k_list, l_T_list = build_game_costs_3p(cost_kind, cost_cfg or {}, D, T)

    # ---------- helpers for stationarity ----------
    def _eq_grad_dot_lam_on(_m, var, k_hint=None, agent=None):
        s = 0.0
        # IC
        if agent == 1 and k_hint == 0:
            for i in _m.S: s += _m.lam_ic1[i] * differentiate(g_ic_expr(_m,1,i), wrt=var, mode=Modes.reverse_symbolic)
        if agent == 2 and k_hint == 0:
            for i in _m.S: s += _m.lam_ic2[i] * differentiate(g_ic_expr(_m,2,i), wrt=var, mode=Modes.reverse_symbolic)
        if agent == 3 and k_hint == 0:
            for i in _m.S: s += _m.lam_ic3[i] * differentiate(g_ic_expr(_m,3,i), wrt=var, mode=Modes.reverse_symbolic)

        # Dynamics rows that touch (k_hint) and (k_hint-1)
        if agent is not None and (k_hint is not None):
            lam_dyn = getattr(_m, f"lam_dyn{agent}")
            if k_hint in _m.Ku:
                for i in _m.S:
                    s += lam_dyn[k_hint, i] * differentiate(g_dyn_expr(_m, agent, k_hint, i), wrt=var, mode=Modes.reverse_symbolic)
            if (k_hint-1) in _m.Ku:
                for i in _m.S:
                    s += lam_dyn[k_hint-1, i] * differentiate(g_dyn_expr(_m, agent, k_hint-1, i), wrt=var, mode=Modes.reverse_symbolic)
        return s

    def _ineq_grad_dot_mu_on(_m, var, k_hint=None):
        if not h_builders: return 0.0
        s = 0.0
        if k_hint is None:
            for h in _m.H:
                for k in _m.Kx:
                    s += _m.mu[h,k] * differentiate(h_builders[int(h)](_m, k), wrt=var, mode=Modes.reverse_symbolic)
            return s
        for h in _m.H:
            s += _m.mu[h,k_hint] * differentiate(h_builders[int(h)](_m, k_hint), wrt=var, mode=Modes.reverse_symbolic)
        return s

    # ---------- stationarity (x,u) for each agent ----------
    def _st_x_rule(agent):
        def r(_m, k, i, _a=agent):
            var = getattr(_m, f"x{_a}")[k, i]
            grad_cost = differentiate(l_k_list[_a-1](_m, k) if k < T else l_T_list[_a-1](_m),
                                     wrt=var, mode=Modes.reverse_symbolic)
            Jg = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=_a)
            Jh = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
            return grad_cost - Jg - Jh == 0
        return r

    def _st_u_rule(agent):
        def r(_m, k, j, _a=agent):
            var = getattr(_m, f"u{_a}")[k, j]
            grad_cost = differentiate(l_k_list[_a-1](_m, k),
                                     wrt=var, mode=Modes.reverse_symbolic)
            Jg = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=_a)
            Jh = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
            return grad_cost - Jg - Jh == 0
        return r

    m.st_x1 = Constraint(m.Kx, m.S, rule=_st_x_rule(1))
    m.st_x2 = Constraint(m.Kx, m.S, rule=_st_x_rule(2))
    m.st_x3 = Constraint(m.Kx, m.S, rule=_st_x_rule(3))
    m.st_u1 = Constraint(m.Ku, m.U, rule=_st_u_rule(1))
    m.st_u2 = Constraint(m.Ku, m.U, rule=_st_u_rule(2))
    m.st_u3 = Constraint(m.Ku, m.U, rule=_st_u_rule(3))

    # ---------- warm start ----------
    for k in m.Ku:
        for j in m.U:
            m.u1[k,j].value = 0.0; m.u2[k,j].value = 0.0; m.u3[k,j].value = 0.0
    for i in m.S:
        m.x1[0,i].value = float(x0_1[i])
        m.x2[0,i].value = float(x0_2[i])
        m.x3[0,i].value = float(x0_3[i])
    # forward-sim seeds
    for k in m.Ku:
        for i in m.S:
            m.x1[k+1,i].value = sum(value(m.A[i,j])*value(m.x1[k,j]) for j in m.S)
            m.x2[k+1,i].value = sum(value(m.A[i,j])*value(m.x2[k,j]) for j in m.S)
            m.x3[k+1,i].value = sum(value(m.A[i,j])*value(m.x3[k,j]) for j in m.S)

    # multipliers
    for i in m.S:
        m.lam_ic1[i].value = 0.0; m.lam_ic2[i].value = 0.0; m.lam_ic3[i].value = 0.0
    for k in m.Ku:
        for i in m.S:
            m.lam_dyn1[k,i].value = 0.0; m.lam_dyn2[k,i].value = 0.0; m.lam_dyn3[k,i].value = 0.0
    if h_builders:
        for h in m.H:
            for k in m.Kx:
                m.mu[h,k].value = 0.0

    return m

def build_h_builders_3(cfg, nx, D):
    """
    Explicit 3-player shared-inequality builders: h(m,k) >= 0.
    Uses ONLY per-player vars: m.x1[k,j], m.x2[k,j], m.x3[k,j].
    Returns: list of callables h(m,k) -> scalar, each enforced for all k in Kx.
    """
    ar = (cfg.get("arena") or {})
    funcs = []

    # strict accessor
    def _x(m, agent, k, j):
        xv = getattr(m, f"x{agent}")
        return xv[k, j]

    # ---------------- Arena keep-in ----------------
    # Box
    if {"xmin","xmax","ymin","ymax"} <= set(ar.keys()):
        xmin, xmax = float(ar["xmin"]), float(ar["xmax"])
        ymin, ymax = float(ar["ymin"]), float(ar["ymax"])
        have_z     = (D == 3) and ("zmin" in ar and "zmax" in ar)
        zmin, zmax = (float(ar["zmin"]), float(ar["zmax"])) if have_z else (None, None)

        for a in (1,2,3):
            funcs.append(lambda m,k,_a=a,_b=xmin: _x(m,_a,k,0) - _b)   # x >= xmin
            funcs.append(lambda m,k,_a=a,_b=xmax: _b - _x(m,_a,k,0))   # xmax - x
            if D >= 2:
                funcs.append(lambda m,k,_a=a,_b=ymin: _x(m,_a,k,1) - _b)
                funcs.append(lambda m,k,_a=a,_b=ymax: _b - _x(m,_a,k,1))
            if have_z:
                funcs.append(lambda m,k,_a=a,_b=zmin: _x(m,_a,k,2) - _b)
                funcs.append(lambda m,k,_a=a,_b=zmax: _b - _x(m,_a,k,2))

    # Sphere
    elif {"cx","cy","cz","r"} <= set(ar.keys()) or ar.get("type") == "sphere":
        cx = float(ar.get("cx", 0.0))
        cy = float(ar.get("cy", 0.0))
        cz = float(ar.get("cz", 0.0))
        R2 = float(ar.get("r", 1.0))**2

        def _sphere_h(agent):
            def h(m,k,_a=agent,_cx=cx,_cy=cy,_cz=cz,_R2=R2):
                px = _x(m,_a,k,0) - _cx
                py = _x(m,_a,k,1) - _cy if D >= 2 else 0.0
                pz = _x(m,_a,k,2) - _cz if D == 3 else 0.0
