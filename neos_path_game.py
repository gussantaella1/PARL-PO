# neos_path_game.py — ONE BIG MCP version (KKT-only; costs can be injected)
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
    import shutil
    exe = (
        path_exe
        or os.environ.get("PATHAMPL")
        or getattr(model, "_pathampl", None)
        or "pathampl"
    )
    resolved = shutil.which(exe) or exe
    if not os.path.isabs(resolved):
        print(f"[PATH] using solver from PATH: {resolved}")
    else:
        if not (os.path.isfile(resolved) and os.access(resolved, os.X_OK)):
            raise FileNotFoundError(f"PATH executable not found or not executable: {resolved}")

    opt = SolverFactory("asl")
    opt.options["solver"] = resolved
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
    # Keep var bounds WIDE; real limits go into shared inequalities (h_builders + boxes below)
    x_var_box=(-1e6, 1e6),
    u_var_box=(-1e3, 1e3),
    # Shared path-inequality builders: list of callables h(m,k) >= 0
    h_builders=None,
    # Optional extra box limits (pushed into h̃ regardless of arena)
    x_lb=None, x_ub=None,   # arrays of length nx (apply to both players)
    u_lb=None, u_ub=None,   # arrays of length nu (apply to both players)
    # Optional goal-region inequalities
    goal=None,              # first D entries used
    goal_r=0.0,             # attacker success radius (r^2 - ||p2-goal||^2 >= 0)
    def_keepout_r=0.0,      # defender keep-out (||p1-goal||^2 - r^2 >= 0)
    # Costs: inject stage/terminal cost callables (Pyomo expr). If None, use LQ fallback.
    l1_k=None, l1_T=None,   # defender costs at stage k (0..T-1) and terminal T
    l2_k=None, l2_T=None,   # attacker costs
    # LQ fallback weights (used only if l?_? not provided)
    q_fallback=1.0, r1_fallback=1.0, r2_fallback=1.0,
):
    """
    Build a single MCP for the 2-player horizon game.

    Primals:     τ = {x1(k),u1(k), x2(k),u2(k)} over k
    Shared eqs:  g̃ = [IC1, dyn1, IC2, dyn2] with free multipliers lam̃
    Shared ineq: h̃ >= 0 with μ >= 0 and μ ⟂ h̃
    Stationarity: ∇f_i + Jg̃^T lam̃ + Jh̃^T μ = 0

    NOTE: Discounting/averaging lives in your external cost builders (game_costs.py).
    This MCP just uses the provided l1_k/l1_T/l2_k/l2_T or a simple LQ fallback.
    """
    m = ConcreteModel()

    # ---------- sets ----------
    m.Kx = RangeSet(0, T)        # T+1 states
    m.Ku = RangeSet(0, T-1)      # T controls
    m.S  = RangeSet(0, nx-1)
    m.U  = RangeSet(0, nu-1)

    # ---------- params ----------
    Ad = np.asarray(Ad, float); Bd = np.asarray(Bd, float)
    x0_1 = np.asarray(x0_1, float); x0_2 = np.asarray(x0_2, float)
    m.A  = Param(m.S, m.S, initialize=lambda _m,i,j: float(Ad[i,j]), within=Reals, mutable=False)
    m.B  = Param(m.S, m.U, initialize=lambda _m,i,j: float(Bd[i,j]), within=Reals, mutable=False)

    # Mutable ICs for RHC
    m.x01 = Param(m.S, initialize=lambda _m,i: float(x0_1[i]), within=Reals, mutable=True)
    m.x02 = Param(m.S, initialize=lambda _m,i: float(x0_2[i]), within=Reals, mutable=True)

    # ---------- primals (wide boxes only) ----------
    xlb_w, xub_w = float(x_var_box[0]), float(x_var_box[1])
    ulb_w, uub_w = float(u_var_box[0]), float(u_var_box[1])
    m.x1 = Var(m.Kx, m.S, bounds=lambda _m,k,i: (xlb_w, xub_w))
    m.x2 = Var(m.Kx, m.S, bounds=lambda _m,k,i: (xlb_w, xub_w))
    m.u1 = Var(m.Ku, m.U, bounds=lambda _m,k,j: (ulb_w, uub_w))
    m.u2 = Var(m.Ku, m.U, bounds=lambda _m,k,j: (ulb_w, uub_w))

    # ---------- shared equality g̃(τ)=0 ----------
    def g_ic1_expr(_m, i): return _m.x1[0, i] - _m.x01[i]
    def g_ic2_expr(_m, i): return _m.x2[0, i] - _m.x02[i]
    def g_dyn1_expr(_m, k, i):
        return _m.x1[k+1, i] - sum(_m.A[i,j]*_m.x1[k,j] for j in _m.S) - sum(_m.B[i,j]*_m.u1[k,j] for j in _m.U)
    def g_dyn2_expr(_m, k, i):
        return _m.x2[k+1, i] - sum(_m.A[i,j]*_m.x2[k,j] for j in _m.S) - sum(_m.B[i,j]*_m.u2[k,j] for j in _m.U)

    m.ic1  = Constraint(m.S,        rule=lambda _m,i: g_ic1_expr(_m,i) == 0)
    m.ic2  = Constraint(m.S,        rule=lambda _m,i: g_ic2_expr(_m,i) == 0)
    m.dyn1 = Constraint(m.Ku, m.S,  rule=lambda _m,k,i: g_dyn1_expr(_m,k,i) == 0)
    m.dyn2 = Constraint(m.Ku, m.S,  rule=lambda _m,k,i: g_dyn2_expr(_m,k,i) == 0)

    # ---------- shared multipliers for g̃ ----------
    m.lam_ic1  = Var(m.S,        domain=Reals)
    m.lam_ic2  = Var(m.S,        domain=Reals)
    m.lam_dyn1 = Var(m.Ku, m.S,  domain=Reals)
    m.lam_dyn2 = Var(m.Ku, m.S,  domain=Reals)

    # ---------- h̃ builders (incoming + boxes + goal-region) ----------
    hb = list(h_builders or [])

    def _x(m, who, k, i):  # who=1|2
        return m.x1[k, i] if who == 1 else m.x2[k, i]
    def _u(m, who, k, j):
        return m.u1[k, j] if who == 1 else m.u2[k, j]

    # (A) State/control boxes (apply for all arenas) if provided
    if x_lb is not None and x_ub is not None:
        x_lb = np.asarray(x_lb, float).ravel(); x_ub = np.asarray(x_ub, float).ravel()
        assert x_lb.size == nx and x_ub.size == nx
        for who in (1, 2):
            for i in range(nx):
                lb_i = float(x_lb[i]); ub_i = float(x_ub[i])
                hb.append(lambda m,k,_w=who,_i=i,_lb=lb_i: _x(m,_w,k,_i) - _lb)  # x - lb ≥ 0
                hb.append(lambda m,k,_w=who,_i=i,_ub=ub_i: _ub - _x(m,_w,k,_i))  # ub - x ≥ 0

    if u_lb is not None and u_ub is not None:
        u_lb = np.asarray(u_lb, float).ravel(); u_ub = np.asarray(u_ub, float).ravel()
        assert u_lb.size == nu and u_ub.size == nu
        for who in (1, 2):
            for j in range(nu):
                lb_j = float(u_lb[j]); ub_j = float(u_ub[j])
                hb.append(lambda m,k,_w=who,_j=j,_lb=lb_j: _u(m,_w,k,_j) - _lb)  # u - lb ≥ 0
                hb.append(lambda m,k,_w=who,_j=j,_ub=ub_j: _ub - _u(m,_w,k,_j))  # ub - u ≥ 0

    # (B) Goal-region inequalities (optional)
    if goal is not None:
        goal = np.asarray(goal, float).reshape(-1)
        assert goal.size >= D
        g_vec = goal[:D].copy()
        if goal_r and goal_r > 0.0:
            def h_goal_attacker(m,k,_g=g_vec,_r2=goal_r**2):
                s = 0.0
                for d in range(D):
                    s += (m.x2[k, d] - _g[d])**2
                return _r2 - s  # ≥ 0 inside attacker success disk/sphere
            hb.append(h_goal_attacker)
        if def_keepout_r and def_keepout_r > 0.0:
            def h_goal_def_keepout(m,k,_g=g_vec,_r2=def_keepout_r**2):
                s = 0.0
                for d in range(D):
                    s += (m.x1[k, d] - _g[d])**2
                return s - _r2      # ≥ 0 outside defender keep-out radius
            hb.append(h_goal_def_keepout)

    # ---------- μ ≥ 0, μ ⟂ h̃ ≥ 0 ----------
    if hb:
        m.H = RangeSet(0, len(hb)-1)
        m.mu = Var(m.H, m.Kx, domain=NonNegativeReals)
        def _h_expr(_m, h, k):
            return hb[int(h)](_m, k)
        m.h_comp = Complementarity(
            m.H, m.Kx,
            rule=lambda _m,h,k: complements(_m.mu[h,k] >= 0, _h_expr(_m,h,k) >= 0)
        )

    # ---------- costs (use injected callables if provided) ----------
    # Fallback simple LQ if not provided
    if l1_k is None or l1_T is None or l2_k is None or l2_T is None:
        q = float(q_fallback); r1 = float(r1_fallback); r2 = float(r2_fallback)
        def l1_k(_m,k): return q*sum(_m.x1[k,i]**2 for i in _m.S) + r1*sum(_m.u1[k,j]**2 for j in _m.U)
        def l2_k(_m,k): return q*sum(_m.x2[k,i]**2 for i in _m.S) + r2*sum(_m.u2[k,j]**2 for j in _m.U)
        def l1_T(_m):  return q*sum(_m.x1[T,i]**2 for i in _m.S)
        def l2_T(_m):  return q*sum(_m.x2[T,i]**2 for i in _m.S)

    # ---------- helper: Jgᵀλ + Jhᵀμ on a given variable ----------
    def _eq_grad_dot_lam_on(_m, var, k_hint=None, agent=None):
        s = 0.0
        if agent == 1 and k_hint == 0:
            for i in _m.S:
                s += _m.lam_ic1[i] * differentiate(g_ic1_expr(_m, i), wrt=var, mode=Modes.reverse_symbolic)
        if agent == 2 and k_hint == 0:
            for i in _m.S:
                s += _m.lam_ic2[i] * differentiate(g_ic2_expr(_m, i), wrt=var, mode=Modes.reverse_symbolic)
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
        # Only couple same time index (like Jacobian sparsity)
        if not hasattr(m, "H"): return 0.0
        s = 0.0
        for h in _m.H:
            s += _m.mu[h,k_hint] * differentiate(hb[int(h)](_m, k_hint), wrt=var, mode=Modes.reverse_symbolic)
        return s

    # ---------- stationarity (correct KKT sign) ----------
    def st_x1_rule(_m, k, i):
        var = _m.x1[k, i]
        grad_cost = differentiate(l1_k(_m, k), wrt=var, mode=Modes.reverse_symbolic) if k < T else \
                    differentiate(l1_T(_m), wrt=var, mode=Modes.reverse_symbolic)
        JgTlam = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=1)
        JhTmu  = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
        return grad_cost + JgTlam + JhTmu == 0

    def st_u1_rule(_m, k, j):
        var = _m.u1[k,j]
        grad_cost  = differentiate(l1_k(_m, k), wrt=var, mode=Modes.reverse_symbolic)
        JgTlam     = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=1)
        JhTmu      = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
        return grad_cost + JgTlam + JhTmu == 0

    def st_x2_rule(_m, k, i):
        var = _m.x2[k, i]
        grad_cost = differentiate(l2_k(_m, k), wrt=var, mode=Modes.reverse_symbolic) if k < T else \
                    differentiate(l2_T(_m), wrt=var, mode=Modes.reverse_symbolic)
        JgTlam = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=2)
        JhTmu  = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
        return grad_cost + JgTlam + JhTmu == 0

    def st_u2_rule(_m, k, j):
        var = _m.u2[k,j]
        grad_cost  = differentiate(l2_k(_m, k), wrt=var, mode=Modes.reverse_symbolic)
        JgTlam     = _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=2)
        JhTmu      = _ineq_grad_dot_mu_on(_m, var, k_hint=k)
        return grad_cost + JgTlam + JhTmu == 0

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
    if hasattr(m, "H"):
        for h in m.H:
            for k in m.Kx:
                m.mu[h,k].value = 0.0

    return m
