# neos_path_game_translation.py — ONE BIG MCP version
import os
import numpy as np
from pyomo.environ import *
from pyomo.mpec import Complementarity, complements
from pyomo.core.expr.calculus.derivatives import differentiate, Modes


# ----------------------- public helpers -----------------------
def extract_trajectories(m):
    Kx = list(m.Kx); Ku = list(m.Ku); S = list(m.S); U = list(m.U)
    X1 = np.array([[value(m.x1[k, i]) for i in S] for k in Kx], dtype=float)
    X2 = np.array([[value(m.x2[k, i]) for i in S] for k in Kx], dtype=float)
    U1 = np.array([[value(m.u1[k, j]) for j in U] for k in Ku], dtype=float)
    U2 = np.array([[value(m.u2[k, j]) for j in U] for k in Ku], dtype=float)
    return X1, U1, X2, U2


def solve_with_local_path(model, path_exe=None, tee=True, path_options=None):
    import shutil, os
    exe = (
        path_exe
        or os.environ.get("PATHAMPL")
        or getattr(model, "_pathampl", None)
        or "pathampl"
    )
    # resolve to absolute if it’s on PATH
    resolved = shutil.which(exe) or exe
    if not os.path.isabs(resolved):
        # let PATH handle it; just a friendly print
        print(f"[PATH] using solver from PATH: {resolved}")
    else:
        if not (os.path.isfile(resolved) and os.access(resolved, os.X_OK)):
            raise FileNotFoundError(f"PATH executable not found or not executable: {resolved}")

    opt = SolverFactory("asl")                # <-- use ASL front-end
    opt.options["solver"] = resolved          # <-- CRUCIAL: tell ASL which binary to run
    # Typical PATH options (tweak as needed)
    for k, v in (path_options or {
        "proximal": 0.01,
        "start": 1,
        "crash": 1,
        "major_iteration_limit": 20000
    }).items():
        opt.options[str(k)] = v

    print(f"[PATH] solver binary: {opt.options['solver']}")
    res = opt.solve(
        model,
        tee=tee,
        keepfiles=True,
        symbolic_solver_labels=True,
        load_solutions=False,
    )
    status = res.solver.status
    term   = res.solver.termination_condition
    print(f"[PATH] status={status}, termination={term}")

    if (str(status).lower() == "ok") and term in (
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
    ):
        model.solutions.load_from(res, default_variable_value=None)
        return res

    raise RuntimeError(f"PATH failed (status={status}, termination={term}).")





# ----------------------- one big MCP builder -----------------------
def build_mcp_two_player_one_shot(
    Ad, Bd, T, nx, nu,
    x0_1, x0_2,
    D=3,
    # Keep variable bounds WIDE; push true limits into h_builders (shared ineq)
    x_var_box=(-1e6, 1e6),
    u_var_box=(-1e3, 1e3),
    # Shared path-inequality builders: list of callables h(m,k) >= 0
    h_builders=None,
    # Cost
    cost_kind="chase_escape_tail",
    cost_cfg=None,
):
    """
    Build a *single* MCP for the 2-player horizon game, mirroring the Julia pattern.

    Primals:     τ = {x1(k),u1(k), x2(k),u2(k)} over k
    Shared eqs:  g̃ = [IC1, dyn1, IC2, dyn2] with free multipliers lam̃
    Shared ineq: h̃ >= 0 with multipliers mũ >= 0 and mũ ⟂ h̃
    Stationarity (for each player's vars): ∇f_i - Jg̃^T lam̃ - Jh̃^T mũ = 0
    """
    cost_cfg = cost_cfg or {}
    m = ConcreteModel()

    # ---------- sets ----------
    m.Kx = RangeSet(0, T)        # state time indices
    m.Ku = RangeSet(0, T-1)      # control time indices
    m.S  = RangeSet(0, nx-1)     # state components
    m.U  = RangeSet(0, nu-1)     # control components

    # ---------- params ----------
    Ad = np.asarray(Ad, float); Bd = np.asarray(Bd, float)
    x0_1 = np.asarray(x0_1, float); x0_2 = np.asarray(x0_2, float)
    m.A  = Param(m.S, m.S, initialize=lambda _m,i,j: float(Ad[i,j]), within=Reals, mutable=False)
    m.B  = Param(m.S, m.U, initialize=lambda _m,i,j: float(Bd[i,j]), within=Reals, mutable=False)

    # Make IC params MUTABLE so RHC can update them each turn
    m.x01 = Param(m.S, initialize=lambda _m,i: float(x0_1[i]), within=Reals, mutable=True)
    m.x02 = Param(m.S, initialize=lambda _m,i: float(x0_2[i]), within=Reals, mutable=True)

    # ---------- primals (wide boxes; do *not* use for real limits) ----------
    xlb, xub = float(x_var_box[0]), float(x_var_box[1])
    ulb, uub = float(u_var_box[0]), float(u_var_box[1])
    m.x1 = Var(m.Kx, m.S, bounds=lambda _m,k,i: (xlb, xub))
    m.x2 = Var(m.Kx, m.S, bounds=lambda _m,k,i: (xlb, xub))
    m.u1 = Var(m.Ku, m.U, bounds=lambda _m,k,j: (ulb, uub))
    m.u2 = Var(m.Ku, m.U, bounds=lambda _m,k,j: (ulb, uub))

    # ---------- shared equality *expressions* g̃(τ) g tilde ----------
    #   ICs
    def g_ic1_expr(_m, i): return _m.x1[0, i] - _m.x01[i]
    def g_ic2_expr(_m, i): return _m.x2[0, i] - _m.x02[i]
    #   dynamics
    def g_dyn1_expr(_m, k, i):
        return _m.x1[k+1, i] - sum(_m.A[i,j]*_m.x1[k,j] for j in _m.S) - sum(_m.B[i,j]*_m.u1[k,j] for j in _m.U)
    def g_dyn2_expr(_m, k, i):
        return _m.x2[k+1, i] - sum(_m.A[i,j]*_m.x2[k,j] for j in _m.S) - sum(_m.B[i,j]*_m.u2[k,j] for j in _m.U)

    # enforce g̃ = 0 as constraints (we also introduce their multipliers below)
    m.ic1  = Constraint(m.S,        rule=lambda _m,i: g_ic1_expr(_m,i) == 0)
    m.ic2  = Constraint(m.S,        rule=lambda _m,i: g_ic2_expr(_m,i) == 0)
    m.dyn1 = Constraint(m.Ku, m.S,  rule=lambda _m,k,i: g_dyn1_expr(_m,k,i) == 0)
    m.dyn2 = Constraint(m.Ku, m.S,  rule=lambda _m,k,i: g_dyn2_expr(_m,k,i) == 0)

    # ---------- shared multipliers for g̃ (free) ----------
    m.lam_ic1  = Var(m.S,        domain=Reals)
    m.lam_ic2  = Var(m.S,        domain=Reals)
    m.lam_dyn1 = Var(m.Ku, m.S,  domain=Reals)
    m.lam_dyn2 = Var(m.Ku, m.S,  domain=Reals)

    # ---------- shared inequalities and μ ≥ 0 with μ ⟂ h ----------
    h_builders = h_builders or []
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
    def _pos(x,k): return [x[k,i] for i in range(D)]
    def _vel(x,k): return [x[k,D+i] for i in range(D)]
    def _dot(a,b): return sum(ai*bi for ai,bi in zip(a,b))
    def _sumsq(v): return sum(vi*vi for vi in v)
    def _norm(v):  return sqrt(_sumsq(v) + 1e-9)

    ck = (cost_kind or "chase_escape_tail").lower()
    if ck == "chase_escape_tail":
        c_eff1  = float(cost_cfg.get("effort_w1",   0.01))
        c_eff2  = float(cost_cfg.get("effort_w2",   0.01))
        c_wfar  = float(cost_cfg.get("w_far",       1.0))
        c_wterm = float(cost_cfg.get("w_term",      1.0))
        c_ddes  = float(cost_cfg.get("follow_gap",  0.6))
        c_wlong = float(cost_cfg.get("w_tail_long", 1.0))
        c_wlat  = float(cost_cfg.get("w_tail_lat",  8.0))
        c_vref  = float(cost_cfg.get("tail_v_ref",  0.2))
        eps_R   = float(cost_cfg.get("path_eps_R",  1e-3))

        def tail_cost(pL, pF, vL):
            speed = _norm(vL); vhat = [vi / (speed + 1e-9) for vi in vL]
            r  = [pF[i]-pL[i] for i in range(D)]
            s  = _dot(r, vhat)               # parallel
            rperp = [r[i] - s*vhat[i] for i in range(D)]
            tail = c_wlong*(s + c_ddes)**2 + c_wlat*_sumsq(rperp)
            blend = (speed*speed)/(speed*speed + c_vref*c_vref)
            return blend*tail + (1.0 - blend)*_sumsq(r)

        def l1_k(_m,k):
            p2 = _pos(_m.x2,k); v2 = _vel(_m.x2,k); p1 = _pos(_m.x1,k)
            u1 = [_m.u1[k,j] for j in _m.U]
            return tail_cost(p2,p1,v2) + (c_eff1 + eps_R)*_sumsq(u1)

        def l2_k(_m,k):
            p1 = _pos(_m.x1,k); p2 = _pos(_m.x2,k)
            u2 = [_m.u2[k,j] for j in _m.U]
            r  = [p2[i]-p1[i] for i in range(D)]
            return -c_wfar*_sumsq(r) + (c_eff2 + eps_R)*_sumsq(u2)

        def l1_T(_m):
            p2 = _pos(_m.x2,T); v2 = _vel(_m.x2,T); p1 = _pos(_m.x1,T)
            return c_wterm*tail_cost(p2,p1,v2)

        def l2_T(_m):
            p1 = _pos(_m.x1,T); p2 = _pos(_m.x2,T)
            r  = [p2[i]-p1[i] for i in range(D)]
            return -c_wterm*_sumsq(r)

    else:
        # Simple LQ fallback (diag Q on states; diag R on inputs)
        q = float(cost_cfg.get("Q", 1.0)); r1 = float(cost_cfg.get("R1", 1.0)); r2 = float(cost_cfg.get("R2", 1.0))
        def l1_k(_m,k): return q*sum(_m.x1[k,i]**2 for i in _m.S) + r1*sum(_m.u1[k,j]**2 for j in _m.U)
        def l2_k(_m,k): return q*sum(_m.x2[k,i]**2 for i in _m.S) + r2*sum(_m.u2[k,j]**2 for j in _m.U)
        def l1_T(_m):  return q*sum(_m.x1[T,i]**2 for i in _m.S)
        def l2_T(_m):  return q*sum(_m.x2[T,i]**2 for i in _m.S)

    # ---------- helper: shared (Jg^T lam + Jh^T mu) on a given variable ----------
    def _eq_grad_dot_lam_on(_m, var, k_hint=None, agent=None):
        s = 0.0
        # IC row where k==0
        if agent == 1 and k_hint == 0:
            for i in _m.S:
                s += _m.lam_ic1[i] * differentiate(g_ic1_expr(_m, i), wrt=var, mode=Modes.reverse_symbolic)
        if agent == 2 and k_hint == 0:
            for i in _m.S:
                s += _m.lam_ic2[i] * differentiate(g_ic2_expr(_m, i), wrt=var, mode=Modes.reverse_symbolic)
        # Dynamics rows – only the two time slots that can involve this var
        if agent == 1 and (k_hint is not None):
            if k_hint in _m.Ku:
                for i in _m.S:
                    s += _m.lam_dyn1[k_hint, i] * differentiate(g_dyn1_expr(_m, k_hint, i), wrt=var, mode=Modes.reverse_symbolic)
            if (k_hint-1) in _m.Ku:
                for i in _m.S:
                    s += _m.lam_dyn1[k_hint-1, i] * differentiate(g_dyn1_expr(_m, k_hint-1, i), wrt=var, mode=Modes.reverse_symbolic)
        if agent == 2 and (k_hint is not None):
            if k_hint in _m.Ku:
                for i in _m.S:
                    s += _m.lam_dyn2[k_hint, i] * differentiate(g_dyn2_expr(_m, k_hint, i), wrt=var, mode=Modes.reverse_symbolic)
            if (k_hint-1) in _m.Ku:
                for i in _m.S:
                    s += _m.lam_dyn2[k_hint-1, i] * differentiate(g_dyn2_expr(_m, k_hint-1, i), wrt=var, mode=Modes.reverse_symbolic)
        return s

    def _ineq_grad_dot_mu_on(_m, var, k_hint=None):
        if not h_builders: return 0.0
        s = 0.0
        if k_hint is None:    # fallback (rare)
            for h in _m.H:
                for k in _m.Kx:
                    s += _m.mu[h,k] * differentiate(h_builders[int(h)](_m, k), wrt=var, mode=Modes.reverse_symbolic)
            return s
        for h in _m.H:
            s += _m.mu[h,k_hint] * differentiate(h_builders[int(h)](_m, k_hint), wrt=var, mode=Modes.reverse_symbolic)
        return s

    # ---------- stationarity for every state & control (both players) ----------
    # x1
    def st_x1_rule(_m, k, i):
        var = _m.x1[k, i]
        if k < T:
            grad_cost = differentiate(l1_k(_m, k), wrt=var, mode=Modes.reverse_symbolic)
        else:
            grad_cost = differentiate(l1_T(_m), wrt=var, mode=Modes.reverse_symbolic)
        JgTlam = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=1)  # <-- pass agent
        JhTmu  = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
        return grad_cost - JgTlam - JhTmu == 0

    # u1
    def st_u1_rule(_m, k, j):
        var = _m.u1[k,j]
        grad_cost  = differentiate(l1_k(_m, k), wrt=var, mode=Modes.reverse_symbolic)
        JgTlam     = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=1)
        JhTmu      = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
        return grad_cost - JgTlam - JhTmu == 0

    # x2
    def st_x2_rule(_m, k, i):
        var = _m.x2[k, i]
        if k < T:
            grad_cost = differentiate(l2_k(_m, k), wrt=var, mode=Modes.reverse_symbolic)
        else:
            grad_cost = differentiate(l2_T(_m), wrt=var, mode=Modes.reverse_symbolic)
        JgTlam = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=2)  # <-- pass agent
        JhTmu  = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
        return grad_cost - JgTlam - JhTmu == 0

    # u2
    def st_u2_rule(_m, k, j):
        var = _m.u2[k,j]
        grad_cost  = differentiate(l2_k(_m, k), wrt=var, mode=Modes.reverse_symbolic)
        JgTlam     = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=2)
        JhTmu      = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
        return grad_cost - JgTlam - JhTmu == 0

    m.st_x1 = Constraint(m.Kx, m.S, rule=st_x1_rule)
    m.st_u1 = Constraint(m.Ku, m.U, rule=st_u1_rule)
    m.st_x2 = Constraint(m.Kx, m.S, rule=st_x2_rule)
    m.st_u2 = Constraint(m.Ku, m.U, rule=st_u2_rule)

    # ---------- warm-starts ----------
    for k in m.Ku:
        for j in m.U:
            m.u1[k,j].value = 0.0
            m.u2[k,j].value = 0.0
    for i in m.S:
        m.x1[0,i].value = float(x0_1[i])
        m.x2[0,i].value = float(x0_2[i])
    for k in m.Ku:
        for i in m.S:
            m.x1[k+1,i].value = sum(value(m.A[i,j]) * value(m.x1[k,j]) for j in m.S)
            m.x2[k+1,i].value = sum(value(m.A[i,j]) * value(m.x2[k,j]) for j in m.S)
    for i in m.S:
        m.lam_ic1[i].value = 0.0; m.lam_ic2[i].value = 0.0
    for k in m.Ku:
        for i in m.S:
            m.lam_dyn1[k,i].value = 0.0
            m.lam_dyn2[k,i].value = 0.0
    if h_builders:
        for h in m.H:
            for k in m.Kx:
                m.mu[h,k].value = 0.0

    return m
