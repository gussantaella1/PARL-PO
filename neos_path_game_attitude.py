# neos_path_game_attitude.py
import numpy as np
import pyomo.environ as pyo
from pyomo.mpec import Complementarity as Comp, complements
from pyomo.core.expr.calculus.derivatives import differentiate, Modes
from pyomo.core.expr.visitor import identify_variables

# ---------------------- helpers (unchanged API) ----------------------

def solve_with_ipopt_qp(model, tee=True, options=None, executable=None):
    opt = pyo.SolverFactory("ipopt", executable=executable) if executable \
          else pyo.SolverFactory("ipopt")
    opt.available(exception_flag=True)
    if options:
        for k, v in options.items():
            opt.options[k] = v
    opt.options.setdefault("tol", 1e-8)
    opt.options.setdefault("constr_viol_tol", 1e-8)
    opt.options.setdefault("max_iter", 2000)
    opt.options.setdefault("linear_solver", "mumps")
    res = opt.solve(model, tee=tee)
    model.solutions.load_from(res, default_variable_value=True)
    return res

def derive_desired_dirs_from_plan(X_self, X_other, align="x"):
    """Return list of T world-frame unit vectors from self->other (LOS)."""
    T = X_self.shape[0]
    D = 3 if X_self.shape[1] >= 6 else 2
    d_seq = []
    for t in range(T):
        ps = np.asarray(X_self[t, :D], float)
        po = np.asarray(X_other[t, :D], float)
        rel = po - ps
        if D == 2:
            rel = np.array([rel[0], rel[1], 0.0])
        n = np.linalg.norm(rel)
        d_seq.append(rel / (n + 1e-12) if n > 1e-12 else np.array([1.0, 0.0, 0.0]))
    return d_seq

def bx_of_q_tuple(q0, q1, q2, q3):
    """Body x-axis (boresight) expressed in WORLD from q=(w,x,y,z)."""
    w, x, y, z = q0, q1, q2, q3
    return (
        1 - 2*(y*y + z*z),
        2*(x*y - z*w),
        2*(x*z + y*w),
    )

# ---------- utility: guard against trivial boolean constraints ----------

def _stationarity_constraint(expr):
    """
    Turn a derivative expression into a valid Pyomo constraint.
    If the expression is constant (no variables), return Constraint.Feasible
    to avoid creating a trivial boolean (True) constraint.
    """
    try:
        vars_in = list(identify_variables(expr, include_fixed=False))
    except Exception:
        vars_in = []
    if len(vars_in) == 0:
        return pyo.Constraint.Feasible
    return expr == 0

# ----------------------- MCP/KKT attitude builder -----------------------

    #     """
    # Two-player quaternion attitude game as ONE MCP (PATH).
    # Each agent i has KKT conditions for its own nonlinear OCP (quat step + rigid-body rates),
    # with box bounds on ω_i and τ_i enforced via complementarity.
    # Costs can couple the players through ctx (e.g., LOS alignment / misalignment).

    # Variables per agent i ∈ {1,2}:
    #   q_i[k,0..3], w_i[k,0..2] for k=0..T-1
    #   s_i[k]≥0 (dq-norm helper) for k=0..T-2
    #   τ_i[k,0..2] for k=0..T-2

    # Equalities (per agent, per stage):
    #   r_w^i[k,:] = 0, r_q^i[k,:] = 0, r_s^i[k] = 0, r_unit^i[k] = 0

    # Complementarity (per agent, per stage):
    #   0 ≤ w_i[k,j]-(-wmax) ⟂ λ^w_lo_i[k,j] ≥ 0
    #   0 ≤  wmax - w_i[k,j] ⟂ λ^w_hi_i[k,j] ≥ 0
    #   0 ≤ τ_i[k,j]-(-taumax) ⟂ λ^t_lo_i[k,j] ≥ 0
    #   0 ≤  taumax - τ_i[k,j] ⟂ λ^t_hi_i[k,j] ≥ 0

    # KKT (Nash): ∂L_i/∂(q_i,w_i,τ_i,s_i)=0 where
    #   L_i = J_i(q1,w1,q2,w2) + multipliers_i ⋅ residuals_i
    # """

def quad3(var_ki, k, weights):
    # var_ki is a 2D Var (e.g., b.dth or b.w)
    return sum(float(weights[i]) * (var_ki[k, i]**2) for i in range(3))

def _stationarity_constraint(expr):
    """Return a valid Pyomo equality for stationarity; skip if constant."""
    try:
        vars_in = list(identify_variables(expr, include_fixed=False))
    except Exception:
        vars_in = []
    if len(vars_in) == 0:
        return pyo.Constraint.Feasible
    return expr == 0

def _ensure_len_T(seq, T, default_vec):
    """Return a list of length T of 3-vectors (np.array), padding/trimming as needed."""
    if seq is None:
        return [np.asarray(default_vec, float).ravel() for _ in range(T)]
    out = [np.asarray(v, float).ravel() for v in seq]
    if len(out) == 0:
        out = [np.asarray(default_vec, float).ravel()]
    if len(out) < T:
        out = out + [out[-1]] * (T - len(out))
    elif len(out) > T:
        out = out[:T]
    return out


def build_mcp_attitude_linear_two_player_qp(
    T: int,
    dt: float,
    Qth=(1.0, 1.0, 1.0),
    Qw=(1e-2, 1e-2, 1e-2),
    Rtau=(1e-3, 1e-3, 1e-3),
    Qdw=(0.0, 0.0, 0.0),
    J1_diag=(12.0, 10.0, 8.0), D1_diag=(0.0, 0.0, 0.0),
    J2_diag=(12.0, 10.0, 8.0), D2_diag=(0.0, 0.0, 0.0),
    dth10=(0.0, 0.0, 0.0), w10=(0.0, 0.0, 0.0),
    dth20=(0.0, 0.0, 0.0), w20=(0.0, 0.0, 0.0),
    wmax=5.0, taumax=5.0,
    d_seq1=None, d_seq2=None,   # kept for compatibility; not used by the cost now
    dth_ref1=None,              # length-T list/array of 3-vectors
    dth_ref2=None,              # length-T list/array of 3-vectors
):
    """
    Linear small-angle, rigid-body attitude game (two agents) as a smooth QP.

    Dynamics (per agent):
        δθ_{k+1} = δθ_k + dt * ω_k
        ω_{k+1}  = ω_k  + dt * J^{-1} ( τ_k - D ω_k )

    Decision vars:
        δθ[k,3], ω[k,3] for k=0..T-1
        τ[k,3]          for k=0..T-2

    Constraints: linear equalities above (for k=0..T-2) + fixed initial state.
    Bounds:      -wmax ≤ ω ≤ wmax, -τmax ≤ τ ≤ τmax.

    Objective:
        ∑_k [ 0.5‖δθ_k - δθ_ref,k‖_{Qth}^2 + 0.5‖ω_k‖_{Qw}^2 ]
      + ∑_{k<T-1} [ 0.5‖τ_k‖_{Rtau}^2 + 0.5‖(ω_{k+1}-ω_k)‖_{Qdw}^2 ]

    Convex if weights ≥ 0 and J-diagonals > 0.
    """
    assert T >= 2, "Need T>=2 for dynamics with control."

    # defaults / shape checks
    if d_seq1 is None: d_seq1 = [np.array([1.0,0.0,0.0])] * T
    if d_seq2 is None: d_seq2 = [np.array([1.0,0.0,0.0])] * T
    assert len(d_seq1) == T and len(d_seq2) == T, "d_seq length must equal T"

    if dth_ref1 is None: dth_ref1 = [np.zeros(3)] * T
    if dth_ref2 is None: dth_ref2 = [np.zeros(3)] * T
    dth_ref1 = [np.asarray(v, float).ravel() for v in dth_ref1]
    dth_ref2 = [np.asarray(v, float).ravel() for v in dth_ref2]
    assert len(dth_ref1) == T and len(dth_ref2) == T, "dth_ref length must equal T"

    # model & sets
    m = pyo.ConcreteModel()
    m.K  = pyo.RangeSet(0, T-1)   # state stages
    m.Ku = pyo.RangeSet(0, T-2)   # control stages
    m.A1 = pyo.Block()
    m.A2 = pyo.Block()
    m.dt = pyo.Param(initialize=float(dt), mutable=False)

    # weights -> tuples of floats
    Qth  = tuple(float(x) for x in Qth)
    Qw   = tuple(float(x) for x in Qw)
    Rtau = tuple(float(x) for x in Rtau)
    Qdw  = tuple(float(x) for x in Qdw)

    J1x,J1y,J1z = map(float, J1_diag)
    D1x,D1y,D1z = map(float, D1_diag)
    J2x,J2y,J2z = map(float, J2_diag)
    D2x,D2y,D2z = map(float, D2_diag)

    wB   = (-float(wmax),   float(wmax))
    tauB = (-float(taumax), float(taumax))

    def _build_agent(b, Jxyz, Dxyz, dth0, w0, dth_ref):
        Jx,Jy,Jz = Jxyz
        Dx,Dy,Dz = Dxyz

        # decision variables
        b.dth = pyo.Var(m.K,  range(3), initialize=0.0)                            # δθ
        b.w   = pyo.Var(m.K,  range(3), bounds=lambda _m,k,i: wB,   initialize=0.0) # ω
        b.tau = pyo.Var(m.Ku, range(3), bounds=lambda _m,k,i: tauB, initialize=0.0) # τ

        # initial conditions (fixed)
        for i, v in enumerate(dth0): b.dth[0, i].fix(float(v))
        for i, v in enumerate(w0):   b.w[0, i].fix(float(v))

        # linear dynamics
        def dth_dyn_rule(_m, k, i):
            if k == T-1: return pyo.Constraint.Skip
            return b.dth[k+1, i] - (b.dth[k, i] + m.dt * b.w[k, i]) == 0
        b.r_dth = pyo.Constraint(m.K, range(3), rule=dth_dyn_rule)

        def w_dyn_rule(_m, k, i):
            if k == T-1: return pyo.Constraint.Skip
            if   i == 0:
                rhs = b.w[k,i] + m.dt * ((b.tau[k,i] - Dx*b.w[k,i]) / Jx)
            elif i == 1:
                rhs = b.w[k,i] + m.dt * ((b.tau[k,i] - Dy*b.w[k,i]) / Jy)
            else:
                rhs = b.w[k,i] + m.dt * ((b.tau[k,i] - Dz*b.w[k,i]) / Jz)
            return b.w[k+1, i] - rhs == 0
        b.r_w = pyo.Constraint(m.K, range(3), rule=w_dyn_rule)

        # stage cost (tracks δθ_ref)
        def _stage_cost(k):
            expr = 0.0
            # tracking & rate penalties
            for i in range(3):
                expr += 0.5 * Qth[i] * (b.dth[k, i] - float(dth_ref[k][i]))**2
                expr += 0.5 * Qw[i]  * (b.w[k, i]**2)
            # effort & smoothness
            if k <= T-2:
                for i in range(3):
                    expr += 0.5 * Rtau[i] * (b.tau[k, i]**2)
                if any(q > 0 for q in Qdw):
                    for i in range(3):
                        expr += 0.5 * Qdw[i] * ((b.w[k+1, i] - b.w[k, i])**2)
            return expr

        b.stage_cost = pyo.Expression(m.K, rule=lambda _m, kk: _stage_cost(kk))

    # build agents (note: d_seq* kept but unused here)
    _build_agent(m.A1, (J1x,J1y,J1z), (D1x,D1y,D1z), dth10, w10, dth_ref1)
    _build_agent(m.A2, (J2x,J2y,J2z), (D2x,D2y,D2z), dth20, w20, dth_ref2)

    # total objective (J1 + J2)
    def _total_cost(_m):
        expr = 0.0
        for k in _m.K:
            expr += _m.A1.stage_cost[k] + _m.A2.stage_cost[k]
        return expr
    m.obj = pyo.Objective(rule=_total_cost, sense=pyo.minimize)

    # metadata
    m.T = T
    m.dt_val = float(dt)
    m.meta = {
        "Qth": Qth, "Qw": Qw, "Rtau": Rtau, "Qdw": Qdw,
        "wmax": float(wmax), "taumax": float(taumax),
    }
    return m
