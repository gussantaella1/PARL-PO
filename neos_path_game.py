# neos_path_game.py — ONE BIG MCP version
import os
import numpy as np
from pyomo.environ import *
from pyomo.mpec import Complementarity, complements
from pyomo.core.expr.calculus.derivatives import differentiate, Modes
from pyomo.environ import value as pyo_value

from game_costs import build_game_costs



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

    # ---------- shared equality *expressions* g̃(τ) ----------
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
    l1_k, l2_k, l1_T, l2_T = build_game_costs(cost_kind, cost_cfg or {}, D, T)

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



# --- NEW: build one big MCP for N players ---
def build_mcp_N_player_one_shot(
    Ad, Bd, T, nx, nu,
    x0_list,                   # list of N arrays, each (nx,)
    N,
    D=3,
    x_var_box=(-1e6, 1e6),
    u_var_box=(-1e3, 1e3),
    h_builders=None,           # list of callables h(m,k) >= 0; may use m.x1, m.x2, ...
    cost_kind="chase_escape_tail",
    cost_cfg=None,
):
    cost_cfg   = dict(cost_cfg or {})
    h_builders = list(h_builders or [])

    # ---------- model & sets ----------
    m    = ConcreteModel()
    N    = int(N); T = int(T); nx = int(nx); nu = int(nu)
    m.P  = RangeSet(1, N)          # players
    m.Kx = RangeSet(0, T)          # state time indices
    m.Ku = RangeSet(0, T-1)        # control time indices
    m.S  = RangeSet(0, nx-1)       # state components
    m.U  = RangeSet(0, nu-1)       # control components

    # ---------- parameters ----------
    Ad = np.asarray(Ad, float); Bd = np.asarray(Bd, float)
    x0_list = [np.asarray(x0, float) for x0 in x0_list]
    assert len(x0_list) == N, "x0_list must have length N"

    m.A = Param(m.S, m.S, initialize=lambda _m,i,j: float(Ad[i,j]), within=Reals, mutable=False)
    m.B = Param(m.S, m.U, initialize=lambda _m,i,j: float(Bd[i,j]), within=Reals, mutable=False)

    # mutable IC params per player
    def _init_x0(_m, p, i):
        return float(x0_list[p-1][i])
    m.x0 = Param(m.P, m.S, initialize=_init_x0, within=Reals, mutable=True)

    # ---------- primals (wide boxes; real limits via h_builders) ----------
    xlb, xub = float(x_var_box[0]), float(x_var_box[1])
    ulb, uub = float(u_var_box[0]), float(u_var_box[1])

    # canonical variables
    m.x = Var(m.P, m.Kx, m.S, bounds=lambda _m,p,k,i: (xlb, xub))
    m.u = Var(m.P, m.Ku, m.U, bounds=lambda _m,p,k,j: (ulb, uub))

    # ---------- BACKWARD-COMPAT ALIASES (x1,u1,x2,u2, ... , x0{p}) ----------
    # These are true alias views; they don’t add variables/constraints.
    for p in range(1, N+1):
        # These create views with indices (k,i) and (k,j)
        setattr(m, f"x{p}",  Reference(m.x[p, :, :]))   # m.xp[k,i]
        setattr(m, f"u{p}",  Reference(m.u[p, :, :]))   # m.up[k,j]
        setattr(m, f"x0{p}", Reference(m.x0[p, :]))     # m.x0p[i]

    # ---------- shared equality expressions g̃ (IC + dynamics for every player) ----------
    def g_ic_expr(_m, p, i):
        return _m.x[p, 0, i] - _m.x0[p, i]

    def g_dyn_expr(_m, p, k, i):
        # x_p(k+1) - A x_p(k) - B u_p(k) = 0
        return _m.x[p, k+1, i] - sum(_m.A[i,j]*_m.x[p, k, j] for j in _m.S) \
                                - sum(_m.B[i,j]*_m.u[p, k, j] for j in _m.U)

    # impose g̃ = 0 as constraints
    m.ic  = Constraint(m.P, m.S,         rule=lambda _m,p,i: g_ic_expr(_m,p,i) == 0)
    m.dyn = Constraint(m.P, m.Ku, m.S,   rule=lambda _m,p,k,i: g_dyn_expr(_m,p,k,i) == 0)

    # ---------- multipliers for g̃ ----------
    m.lam_ic  = Var(m.P, m.S,       domain=Reals)
    m.lam_dyn = Var(m.P, m.Ku, m.S, domain=Reals)

    # ---------- shared inequalities and μ ≥ 0 with μ ⟂ h ----------
    if h_builders:
        m.H  = RangeSet(0, len(h_builders)-1)
        m.mu = Var(m.H, m.Kx, domain=NonNegativeReals)

        def _h_expr_wrap(_m, h, k):
            # Existing h_builders can keep using m.x1, m.x2, ... thanks to References.
            return h_builders[int(h)](_m, k)

        m.h_comp = Complementarity(
            m.H, m.Kx,
            rule=lambda _m,h,k: complements(_m.mu[h,k] >= 0, _h_expr_wrap(_m,h,k) >= 0)
        )

    # ---------- costs per player ----------
    def _costs_for_N(cost_kind, cost_cfg, D, T, N):
        # Prefer user N-player factory if present
        if "build_game_costs_N" in globals():
            return build_game_costs_N(cost_kind, cost_cfg or {}, D, T, N)
        # Fallback: replicate your 2p cost
        try:
            l1_k, l2_k, l1_T, l2_T = build_game_costs(cost_kind, cost_cfg or {}, D, T)
        except Exception:
            def z_k(_m,_k): return 0.0
            def z_T(_m):   return 0.0
            return ({p:z_k for p in range(1,N+1)},
                    {p:z_T for p in range(1,N+1)})
        replicate_second = bool(cost_cfg.get("replicate_second_cost_for_others", True))
        l_k, l_T = {}, {}
        for p in range(1, N+1):
            if p == 1:
                l_k[p], l_T[p] = l1_k, l1_T
            elif replicate_second:
                l_k[p], l_T[p] = l2_k, l2_T
            else:
                l_k[p] = (lambda _m,_k: 0.0)
                l_T[p] = (lambda _m: 0.0)
        return l_k, l_T

    l_k, l_T = _costs_for_N(cost_kind, cost_cfg, D, T, N)

    # ---------- helpers: Jg^T λ and Jh^T μ on a variable ----------
    def _eq_grad_dot_lam_on(_m, var, k_hint=None, p_hint=None):
        s = 0.0
        if (p_hint is not None) and (k_hint == 0):
            for i in _m.S:
                s += _m.lam_ic[p_hint, i] * differentiate(g_ic_expr(_m, p_hint, i),
                                                          wrt=var, mode=Modes.reverse_symbolic)
        if (p_hint is not None) and (k_hint is not None):
            if k_hint in _m.Ku:
                for i in _m.S:
                    s += _m.lam_dyn[p_hint, k_hint, i] * differentiate(
                        g_dyn_expr(_m, p_hint, k_hint, i),
                        wrt=var, mode=Modes.reverse_symbolic
                    )
            if (k_hint-1) in _m.Ku:
                for i in _m.S:
                    s += _m.lam_dyn[p_hint, k_hint-1, i] * differentiate(
                        g_dyn_expr(_m, p_hint, k_hint-1, i),
                        wrt=var, mode=Modes.reverse_symbolic
                    )
        return s

    def _ineq_grad_dot_mu_on(_m, var, k_hint=None):
        if not h_builders: return 0.0
        s = 0.0
        if k_hint is None:
            for h in _m.H:
                for k in _m.Kx:
                    s += _m.mu[h,k] * differentiate(h_builders[int(h)](_m, k),
                                                    wrt=var, mode=Modes.reverse_symbolic)
            return s
        for h in _m.H:
            s += _m.mu[h,k_hint] * differentiate(h_builders[int(h)](_m, k_hint),
                                                 wrt=var, mode=Modes.reverse_symbolic)
        return s

    # ---------- stationarity ----------
    def st_x_rule(_m, p, k, i):
        var = _m.x[p, k, i]
        grad_cost = differentiate(l_k[p](_m, k), wrt=var, mode=Modes.reverse_symbolic) if k < T \
                    else differentiate(l_T[p](_m),   wrt=var, mode=Modes.reverse_symbolic)
        return grad_cost - _eq_grad_dot_lam_on(_m, var, k_hint=k, p_hint=p) \
                         - _ineq_grad_dot_mu_on(_m, var, k_hint=k) == 0

    def st_u_rule(_m, p, k, j):
        var = _m.u[p, k, j]
        grad_cost = differentiate(l_k[p](_m, k), wrt=var, mode=Modes.reverse_symbolic)
        return grad_cost - _eq_grad_dot_lam_on(_m, var, k_hint=k, p_hint=p) \
                         - _ineq_grad_dot_mu_on(_m, var, k_hint=k) == 0

    m.st_x = Constraint(m.P, m.Kx, m.S, rule=st_x_rule)
    m.st_u = Constraint(m.P, m.Ku, m.U, rule=st_u_rule)

    # ---------- warm-starts ----------
    for p in m.P:
        for k in m.Ku:
            for j in m.U:
                m.u[p, k, j].value = 0.0
    for p in m.P:
        for i in m.S:
            m.x[p, 0, i].value = float(value(m.x0[p, i]))
    for p in m.P:
        for k in m.Ku:
            for i in m.S:
                m.x[p, k+1, i].value = sum(value(m.A[i,j]) * value(m.x[p, k, j]) for j in m.S)
    for p in m.P:
        for i in m.S:
            m.lam_ic[p, i].value = 0.0
    for p in m.P:
        for k in m.Ku:
            for i in m.S:
                m.lam_dyn[p, k, i].value = 0.0
    if h_builders:
        for h in m.H:
            for k in m.Kx:
                m.mu[h, k].value = 0.0

    return m