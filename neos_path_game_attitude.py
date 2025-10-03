# neos_path_game_attitude.py
# ----------------------- ATTITUDE: one big MCP builder -----------------------
import numpy as np
from pyomo.environ import *
from pyomo.mpec import Complementarity, complements
from pyomo.core.expr.calculus.derivatives import differentiate, Modes

# ---- attitude inequality builders (h >= 0) for the MCP ----
def make_attitude_h_builders(
    T,
    thmax,                 # max small-angle magnitude (rad)
    wmax,                  # max body rate (rad/s)
    taumax,                # max torque (Nm)
    which_agents=(1, 2),   # tuple subset of {1,2}
):
    """
    Returns a list of callables h(m,k) >= 0 for PATH.
    Compatible with build_mcp_attitude_two_player_one_shot_linear(), which uses:
        m.A1.x[k,i], m.A1.u[k,j]
        m.A2.x[k,i], m.A2.u[k,j]
    State layout: x[k,0:3]=δθ, x[k,3:5]=ω. Controls: u = τ.
    """
    th2 = float(thmax)**2
    wmax = float(wmax)
    taumax = float(taumax)

    def _th_ball(agent):
        # ||δθ||^2 <= thmax^2  ->  thmax^2 - sum(δθ_i^2) >= 0
        if agent == 1:
            return lambda m, k: th2 - sum(m.A1.x[k, i]**2 for i in range(0, 3))
        else:
            return lambda m, k: th2 - sum(m.A2.x[k, i]**2 for i in range(0, 3))

    def _w_box_hi(agent, ax):  #  wmax - w >= 0
        if agent == 1:
            return lambda m, k: wmax - m.A1.x[k, 3 + ax]
        else:
            return lambda m, k: wmax - m.A2.x[k, 3 + ax]

    def _w_box_lo(agent, ax):  #  wmax + w >= 0  (i.e., w >= -wmax)
        if agent == 1:
            return lambda m, k: wmax + m.A1.x[k, 3 + ax]
        else:
            return lambda m, k: wmax + m.A2.x[k, 3 + ax]

    def _tau_box_hi(agent, ax):  #  taumax - tau >= 0
        if agent == 1:
            return lambda m, k: (taumax - m.A1.u[k, ax]) if k < T else 1.0
        else:
            return lambda m, k: (taumax - m.A2.u[k, ax]) if k < T else 1.0

    def _tau_box_lo(agent, ax):  #  taumax + tau >= 0  (i.e., tau >= -taumax)
        if agent == 1:
            return lambda m, k: (taumax + m.A1.u[k, ax]) if k < T else 1.0
        else:
            return lambda m, k: (taumax + m.A2.u[k, ax]) if k < T else 1.0

    HB = []
    for ag in which_agents:
        # δθ ball
        HB.append(_th_ball(ag))
        # ω box (3 axes, both sides)
        for ax in range(3):
            HB.append(_w_box_hi(ag, ax))
            HB.append(_w_box_lo(ag, ax))
        # τ box (3 axes, both sides) — guard k==T
        for ax in range(3):
            HB.append(_tau_box_hi(ag, ax))
            HB.append(_tau_box_lo(ag, ax))
    return HB


def make_attitude_h_builders_split(T, thmax, wmax, taumax, which_agents=(1,2)):
    """
    Build complementarity-friendly shared inequality builders for the *flat* attitude MCP:
      - State-stage (Kx):  δθ-ball and ω-box  (use m.x1/m.x2 at indices 0..2 and 3..5)
      - Control-stage (Ku): τ-box            (use m.u1/m.u2 at indices 0..2)

    Each builder is a callable h(m, k) -> Pyomo expression >= 0.
    We return (HBx, HBu) so you can pass them to:
        build_mcp_attitude_two_player_one_shot_linear(..., h_builders_kx=HBx, h_builders_ku=HBu, ...)
    """
    import numpy as np

    th2 = float(thmax)**2           # bound on ||δθ||^2
    wB  = float(wmax)
    tB  = float(taumax)
    eps = 0.0                       # keep 0.0 unless you need slack; PATH likes crisp bounds

    HBx = []   # builders for state stages k ∈ Kx = {0..T}
    HBu = []   # builders for control stages k ∈ Ku = {0..T-1}

    # ---- helper lambdas that use FLAT variables (NO A1/A2 blocks) ----
    # δθ-ball: thmax^2 - (δθ_x^2 + δθ_y^2 + δθ_z^2) >= 0
    def _h_th_ball_agent(agent):
        if agent == 1:
            return lambda m, k: th2 - (m.x1[k,0]**2 + m.x1[k,1]**2 + m.x1[k,2]**2) + eps
        else:
            return lambda m, k: th2 - (m.x2[k,0]**2 + m.x2[k,1]**2 + m.x2[k,2]**2) + eps

    # ω box as two-sided inequalities split into two complementarity functions per axis:
    #   wB - ω_i >= 0   and   (wB + ω_i) >= 0
    # (PATH wants h >= 0; two constraints per axis are fine.)
    def _h_w_hi_agent(agent, i):
        if agent == 1:
            return lambda m, k: (wB - m.x1[k, 3+i]) + eps
        else:
            return lambda m, k: (wB - m.x2[k, 3+i]) + eps

    def _h_w_lo_agent(agent, i):
        if agent == 1:
            return lambda m, k: (wB + m.x1[k, 3+i]) + eps
        else:
            return lambda m, k: (wB + m.x2[k, 3+i]) + eps

    # τ box on control stages (k ∈ Ku)
    def _h_tau_hi_agent(agent, i):
        if agent == 1:
            return lambda m, k: (tB - m.u1[k, i]) + eps
        else:
            return lambda m, k: (tB - m.u2[k, i]) + eps

    def _h_tau_lo_agent(agent, i):
        if agent == 1:
            return lambda m, k: (tB + m.u1[k, i]) + eps
        else:
            return lambda m, k: (tB + m.u2[k, i]) + eps

    # ---- assemble lists for whichever agents you include ----
    for ag in which_agents:
        # state-stage builders (apply on Kx)
        HBx.append(_h_th_ball_agent(ag))        # ||δθ|| <= thmax
        for i in range(3):                      # ω bounds per axis
            HBx.append(_h_w_hi_agent(ag, i))    #  wB - ω_i >= 0
            HBx.append(_h_w_lo_agent(ag, i))    #  wB + ω_i >= 0

        # control-stage builders (apply on Ku)
        for i in range(3):                      # τ bounds per axis
            HBu.append(_h_tau_hi_agent(ag, i))  #  tB - τ_i >= 0
            HBu.append(_h_tau_lo_agent(ag, i))  #  tB + τ_i >= 0

    return HBx, HBu



def _mk_AB_att(dt, J_diag, D_diag):
    """Build small-angle attitude (δθ, ω) linear discrete A(6x6), B(6x3)."""
    Jx,Jy,Jz = map(float, J_diag)
    Dx,Dy,Dz = map(float, D_diag)
    Jinv = np.diag([1.0/Jx, 1.0/Jy, 1.0/Jz])
    Dmat = np.diag([Dx, Dy, Dz])
    I3  = np.eye(3)
    A11 = I3
    A12 = dt * I3
    A21 = np.zeros((3,3))
    A22 = I3 - dt * (Jinv @ Dmat)
    A   = np.block([[A11, A12],
                    [A21, A22]])
    B   = np.vstack([np.zeros((3,3)), dt * Jinv])
    return A, B

def build_mcp_attitude_two_player_one_shot_linear(
    Ad, Bd, T, nx, nu,
    x0_1, x0_2,
    x_var_box=(-1e6, 1e6),
    u_var_box=(-1e3, 1e3),
    # NEW: separate optional lists of builders
    # each builder must be a callable:  h(m, k)  -> expression >= 0
    h_builders_kx=None,       # applied on state stages k ∈ Kx (0..T)
    h_builders_ku=None,       # applied on control stages k ∈ Ku (0..T-1)
    # BACK-COMP: if provided (legacy), we treat as Kx-builders
    h_builders=None,
    cost_kind="track_ref",
    cost_cfg=None,
):
    """
    Linear two-player attitude (small-angle) game as one MCP (PATH).
    Supports shared inequalities on both state stages (Kx) and control stages (Ku):
        0 <= mu_x ⟂ h_x(m,k_x) >= 0   for k_x in Kx
        0 <= mu_u ⟂ h_u(m,k_u) >= 0   for k_u in Ku
    If only the legacy `h_builders` is passed, it is applied on Kx.
    """
    cost_cfg = cost_cfg or {}
    m = ConcreteModel()

    # ---------- sets ----------
    m.Kx = RangeSet(0, T)        # state time indices
    m.Ku = RangeSet(0, T-1)      # control time indices
    m.S  = RangeSet(0, nx-1)     # state components (δθ, ω → typically 6)
    m.U  = RangeSet(0, nu-1)     # control components (torque → 3)

    # ---------- params ----------
    Ad = np.asarray(Ad, float); Bd = np.asarray(Bd, float)
    x0_1 = np.asarray(x0_1, float); x0_2 = np.asarray(x0_2, float)
    m.A  = Param(m.S, m.S, initialize=lambda _m,i,j: float(Ad[i,j]), within=Reals, mutable=False)
    m.B  = Param(m.S, m.U, initialize=lambda _m,i,j: float(Bd[i,j]), within=Reals, mutable=False)

    # Mutable ICs for RHC
    m.x01 = Param(m.S, initialize=lambda _m,i: float(x0_1[i]), within=Reals, mutable=True)
    m.x02 = Param(m.S, initialize=lambda _m,i: float(x0_2[i]), within=Reals, mutable=True)

    # ---------- primals ----------
    xlb, xub = float(x_var_box[0]), float(x_var_box[1])
    ulb, uub = float(u_var_box[0]), float(u_var_box[1])
    m.x1 = Var(m.Kx, m.S, bounds=lambda _m,k,i: (xlb, xub))
    m.x2 = Var(m.Kx, m.S, bounds=lambda _m,k,i: (xlb, xub))
    m.u1 = Var(m.Ku, m.U, bounds=lambda _m,k,j: (ulb, uub))
    m.u2 = Var(m.Ku, m.U, bounds=lambda _m,k,j: (ulb, uub))

    # ---------- shared equalities ----------
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

    # multipliers for equalities
    m.lam_ic1  = Var(m.S,        domain=Reals)
    m.lam_ic2  = Var(m.S,        domain=Reals)
    m.lam_dyn1 = Var(m.Ku, m.S,  domain=Reals)
    m.lam_dyn2 = Var(m.Ku, m.S,  domain=Reals)

    # ---------- shared inequalities (Kx and Ku) ----------
    # Back-compat: if legacy h_builders is given and new lists are None, treat as Kx constraints
    if (h_builders is not None) and (h_builders_kx is None) and (h_builders_ku is None):
        h_builders_kx = h_builders

    h_builders_kx = h_builders_kx or []
    h_builders_ku = h_builders_ku or []

    if h_builders_kx:
        m.Hx  = RangeSet(0, len(h_builders_kx)-1)
        m.mu_x = Var(m.Hx, m.Kx, domain=NonNegativeReals)
        def _hx_expr(_m, h, k):
            # builder reads the model (x1/x2/u1/u2) and must return ≥ 0
            return h_builders_kx[int(h)](_m, int(k))
        m.hx_comp = Complementarity(
            m.Hx, m.Kx,
            rule=lambda _m,h,k: complements(_m.mu_x[h,k] >= 0, _hx_expr(_m,h,k) >= 0)
        )

    if h_builders_ku:
        m.Hu  = RangeSet(0, len(h_builders_ku)-1)
        m.mu_u = Var(m.Hu, m.Ku, domain=NonNegativeReals)
        def _hu_expr(_m, h, k):
            return h_builders_ku[int(h)](_m, int(k))
        m.hu_comp = Complementarity(
            m.Hu, m.Ku,
            rule=lambda _m,h,k: complements(_m.mu_u[h,k] >= 0, _hu_expr(_m,h,k) >= 0)
        )

    # ---------- costs (same pattern as translation) ----------
    th_idx = [0,1,2]           # δθ indices
    w_idx  = [3,4,5]           # ω indices

    # helpers
    def _subvec(x,k,idxs): return [x[k,i] for i in idxs]
    def _quad(v, W):          return sum(W[i]*(v[i]**2) for i in range(len(v)))

    ck = (cost_kind or "track_ref").lower()
    if ck == "track_ref":
        Qth = tuple(float(x) for x in cost_cfg.get("Qth", (2.0,2.0,2.0)))
        Qw  = tuple(float(x) for x in cost_cfg.get("Qw",  (1e-2,1e-2,1e-2)))
        R   = tuple(float(x) for x in cost_cfg.get("Rtau",(1e-3,1e-3,1e-3)))
        Qdw = tuple(float(x) for x in cost_cfg.get("Qdw", (0.0,0.0,0.0)))
        dref1 = [np.asarray(v, float).ravel() for v in cost_cfg.get("dth_ref1")]
        dref2 = [np.asarray(v, float).ravel() for v in cost_cfg.get("dth_ref2")]

        def l1_k(_m,k):
            dth = _subvec(_m.x1, k, th_idx); w = _subvec(_m.x1, k, w_idx)
            u   = [ _m.u1[k,j] for j in _m.U ] if k in _m.Ku else [0.0]*len(list(_m.U))
            cost = 0.0
            cost += 0.5*sum(Qth[i]*(dth[i]-float(dref1[k][i]))**2 for i in range(3))
            cost += 0.5*sum(Qw[i]*(w[i]**2) for i in range(3))
            if k in _m.Ku:
                cost += 0.5*sum(R[i]*(u[i]**2) for i in range(3))
                if any(q>0 for q in Qdw):
                    wnext = _subvec(_m.x1, k+1, w_idx)
                    cost += 0.5*sum(Qdw[i]*((wnext[i]-w[i])**2) for i in range(3))
            return cost

        def l2_k(_m,k):
            dth = _subvec(_m.x2, k, th_idx); w = _subvec(_m.x2, k, w_idx)
            u   = [ _m.u2[k,j] for j in _m.U ] if k in _m.Ku else [0.0]*len(list(_m.U))
            cost = 0.0
            cost += 0.5*sum(Qth[i]*(dth[i]-float(dref2[k][i]))**2 for i in range(3))
            cost += 0.5*sum(Qw[i]*(w[i]**2) for i in range(3))
            if k in _m.Ku:
                cost += 0.5*sum(R[i]*(u[i]**2) for i in range(3))
                if any(q>0 for q in Qdw):
                    wnext = _subvec(_m.x2, k+1, w_idx)
                    cost += 0.5*sum(Qdw[i]*((wnext[i]-w[i])**2) for i in range(3))
            return cost

        def l1_T(_m):  # no terminal ref right now; reuse stage form
            return 0.0
        def l2_T(_m):
            return 0.0

    else:
        # minimal LQ fallback
        q = float(cost_cfg.get("Q", 1.0))
        r1 = float(cost_cfg.get("R1", 1.0))
        r2 = float(cost_cfg.get("R2", 1.0))
        def l1_k(_m,k): return q*sum(_m.x1[k,i]**2 for i in _m.S) + r1*sum(_m.u1[k,j]**2 for j in _m.U)
        def l2_k(_m,k): return q*sum(_m.x2[k,i]**2 for i in _m.S) + r2*sum(_m.u2[k,j]**2 for j in _m.U)
        def l1_T(_m):  return q*sum(_m.x1[T,i]**2 for i in _m.S)
        def l2_T(_m):  return q*sum(_m.x2[T,i]**2 for i in _m.S)

    # ---------- gradient helpers (same pattern as translation) ----------
    def _eq_grad_dot_lam_on(_m, var, k_hint=None, agent=None):
        s = 0.0
        # IC row
        if agent == 1 and k_hint == 0:
            for i in _m.S:
                s += _m.lam_ic1[i] * differentiate(g_ic1_expr(_m, i), wrt=var, mode=Modes.reverse_symbolic)
        if agent == 2 and k_hint == 0:
            for i in _m.S:
                s += _m.lam_ic2[i] * differentiate(g_ic2_expr(_m, i), wrt=var, mode=Modes.reverse_symbolic)
        # dyn rows (local in time)
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
        s = 0.0
        # Kx comps
        if hasattr(_m, "Hx") and (k_hint in _m.Kx):
            for h in _m.Hx:
                s += _m.mu_x[h,k_hint] * differentiate(h_builders_kx[int(h)](_m, k_hint), wrt=var, mode=Modes.reverse_symbolic)
        # Ku comps
        if hasattr(_m, "Hu") and (k_hint in _m.Ku):
            for h in _m.Hu:
                s += _m.mu_u[h,k_hint] * differentiate(h_builders_ku[int(h)](_m, k_hint), wrt=var, mode=Modes.reverse_symbolic)
        return s

    # ---------- stationarity ----------
    def st_x1_rule(_m, k, i):
        var = _m.x1[k, i]
        grad_cost = differentiate(l1_k(_m, k) if k < T else l1_T(_m), wrt=var, mode=Modes.reverse_symbolic)
        return (grad_cost - _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=1) - _ineq_grad_dot_mu_on(_m, var, k_hint=k)) == 0

    def st_u1_rule(_m, k, j):
        var = _m.u1[k,j]
        grad_cost = differentiate(l1_k(_m, k), wrt=var, mode=Modes.reverse_symbolic)
        return (grad_cost - _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=1) - _ineq_grad_dot_mu_on(_m, var, k_hint=k)) == 0

    def st_x2_rule(_m, k, i):
        var = _m.x2[k, i]
        grad_cost = differentiate(l2_k(_m, k) if k < T else l2_T(_m), wrt=var, mode=Modes.reverse_symbolic)
        return (grad_cost - _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=2) - _ineq_grad_dot_mu_on(_m, var, k_hint=k)) == 0

    def st_u2_rule(_m, k, j):
        var = _m.u2[k,j]
        grad_cost = differentiate(l2_k(_m, k), wrt=var, mode=Modes.reverse_symbolic)
        return (grad_cost - _eq_grad_dot_lam_on(_m, var, k_hint=k, agent=2) - _ineq_grad_dot_mu_on(_m, var, k_hint=k)) == 0

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

    # zero multipliers
    for i in m.S:
        m.lam_ic1[i].value = 0.0; m.lam_ic2[i].value = 0.0
    for k in m.Ku:
        for i in m.S:
            m.lam_dyn1[k,i].value = 0.0
            m.lam_dyn2[k,i].value = 0.0
    if hasattr(m, "Hx"):
        for h in m.Hx:
            for k in m.Kx:
                m.mu_x[h,k].value = 0.0
    if hasattr(m, "Hu"):
        for h in m.Hu:
            for k in m.Ku:
                m.mu_u[h,k].value = 0.0

    return m
