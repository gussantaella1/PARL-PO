# =============================================
# file: assignment5/parametric_game.py
# =============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
import sympy as sp

@dataclass
class ParametricGame:
    """Python analogue of Julia's ParametricGame.

    Lists have length N (players):
    - objectives[i](x_blocks, theta) -> sympy scalar f_i
    - equality_constraints[i](x_blocks, theta) -> sympy vector g_i
    - inequality_constraints[i](x_blocks, theta) -> sympy vector h_i

    Shared constraints accept (x_blocks, theta).
    """

    objectives: List[Callable[[List[sp.Matrix], sp.Matrix], sp.Expr]]
    equality_constraints: List[Callable[[List[sp.Matrix], sp.Matrix], sp.Matrix]]
    inequality_constraints: List[Callable[[List[sp.Matrix], sp.Matrix], sp.Matrix]]
    shared_equality_constraint: Callable[[List[sp.Matrix], sp.Matrix], sp.Matrix]
    shared_inequality_constraint: Callable[[List[sp.Matrix], sp.Matrix], sp.Matrix]

    parameter_dimension: int
    primal_dimensions: List[int]
    equality_dimensions: List[int]
    inequality_dimensions: List[int]
    shared_equality_dimension: int
    shared_inequality_dimension: int

    # derived
    z_sym: Optional[sp.Matrix] = None
    theta_sym: Optional[sp.Matrix] = None
    F_sym: Optional[sp.Matrix] = None
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None

    # (kept for reference / potential downstream use)
    _x_block_slices: Optional[List[slice]] = None
    _lam_block_slices: Optional[List[slice]] = None
    _mu_block_slices: Optional[List[slice]] = None
    _lam_tilde_slice: Optional[slice] = None
    _mu_tilde_slice: Optional[slice] = None

    def __post_init__(self) -> None:
        N = len(self.objectives)
        assert N == len(self.equality_constraints) == len(self.inequality_constraints)
        assert N == len(self.primal_dimensions) == len(self.equality_dimensions) == len(
            self.inequality_dimensions
        )

        total_dim = (
            sum(self.primal_dimensions)
            + sum(self.equality_dimensions)
            + sum(self.inequality_dimensions)
            + self.shared_equality_dimension
            + self.shared_inequality_dimension
        )

        # decision vector z and block views (x, λ, μ, λ~, μ~)
        z = sp.Matrix(sp.symbols(f"z0:{total_dim}"))
        self.z_sym = z

        x_slices: List[slice] = []
        lam_slices: List[slice] = []
        mu_slices: List[slice] = []

        idx = 0
        for d in self.primal_dimensions:
            x_slices.append(slice(idx, idx + d)); idx += d
        for d in self.equality_dimensions:
            lam_slices.append(slice(idx, idx + d)); idx += d
        for d in self.inequality_dimensions:
            mu_slices.append(slice(idx, idx + d)); idx += d
        lam_tilde_idx = slice(idx, idx + self.shared_equality_dimension); idx += self.shared_equality_dimension
        mu_tilde_idx = slice(idx, idx + self.shared_inequality_dimension)

        x_blocks = [z[s, :] for s in x_slices]
        lam_blocks = [z[s, :] for s in lam_slices]
        mu_blocks = [z[s, :] for s in mu_slices]
        lam_tilde = z[lam_tilde_idx, :]
        mu_tilde = z[mu_tilde_idx, :]

        self._x_block_slices = x_slices
        self._lam_block_slices = lam_slices
        self._mu_block_slices = mu_slices
        self._lam_tilde_slice = lam_tilde_idx
        self._mu_tilde_slice = mu_tilde_idx

        # parameters
        theta = sp.Matrix(sp.symbols(f"theta0:{self.parameter_dimension}"))
        self.theta_sym = theta

        # per-player pieces
        f_list, g_list, h_list = [], [], []
        for i in range(N):
            fi = self.objectives[i](x_blocks, theta)
            gi = self.equality_constraints[i](x_blocks, theta)
            hi = self.inequality_constraints[i](x_blocks, theta)
            if not isinstance(gi, sp.Matrix): gi = sp.Matrix(gi)
            if not isinstance(hi, sp.Matrix): hi = sp.Matrix(hi)
            f_list.append(fi); g_list.append(gi); h_list.append(hi)

        # shared
        g_tilde = self.shared_equality_constraint(x_blocks, theta)
        h_tilde = self.shared_inequality_constraint(x_blocks, theta)
        if not isinstance(g_tilde, sp.Matrix): g_tilde = sp.Matrix(g_tilde)
        if not isinstance(h_tilde, sp.Matrix): h_tilde = sp.Matrix(h_tilde)

        # L_i and stationarity
        grad_blocks = []
        for i in range(N):
            xi = x_blocks[i]; li = lam_blocks[i]; mi = mu_blocks[i]
            L_i = f_list[i] - (li.T * g_list[i])[0] - (mi.T * h_list[i])[0]
            L_i -= (lam_tilde.T * g_tilde)[0] + (mu_tilde.T * h_tilde)[0]
            dLdx_i = sp.Matrix([sp.diff(L_i, xi[k, 0]) for k in range(xi.shape[0])])
            grad_blocks.append(dLdx_i)

        F = sp.Matrix.vstack(*grad_blocks, *g_list, *h_list, g_tilde, h_tilde)
        self.F_sym = F

        # bounds: x, λ, λ~ free; μ, μ~ ≥ 0
        lb = np.concatenate([
            -np.inf * np.ones(sum(self.primal_dimensions)),
            -np.inf * np.ones(sum(self.equality_dimensions)),
            np.zeros(sum(self.inequality_dimensions)),
            -np.inf * np.ones(self.shared_equality_dimension),
            np.zeros(self.shared_inequality_dimension),
        ])
        ub = np.inf * np.ones(total_dim)
        self.lower = lb; self.upper = ub

    def total_dim(self) -> int:
        return (
            sum(self.primal_dimensions)
            + sum(self.equality_dimensions)
            + sum(self.inequality_dimensions)
            + self.shared_equality_dimension
            + self.shared_inequality_dimension
        )


# ---------- PATH (Pyomo+MPEC) backend ----------
from pyomo.environ import (
    ConcreteModel, Var, Reals, ConstraintList, Objective, minimize, value, Constraint
)
from pyomo.mpec import Complementarity, complements
from pyomo.opt import SolverFactory
import pyomo.environ as pyo


# --- SymPy -> Pyomo translator ---
def _sympy_to_pyomo(expr, pvars):
    # Python numbers
    if isinstance(expr, (int, float)):
        return float(expr)
    # SymPy numbers (Integer, Float, Rational, etc.)
    if isinstance(expr, sp.Number):
        if getattr(expr, "is_Integer", False):
            return int(expr)
        return float(expr)
    # Symbol
    if isinstance(expr, sp.Symbol):
        return pvars[expr]
    # Matrix-like must be indexed before convert
    if isinstance(expr, sp.MatrixBase):
        raise TypeError("SymPy Matrix not expected here; index before convert.")
    # Basic ops
    if isinstance(expr, sp.Add):
        return sum(_sympy_to_pyomo(a, pvars) for a in expr.args)
    if isinstance(expr, sp.Mul):
        out = 1
        for a in expr.args:
            out = out * _sympy_to_pyomo(a, pvars)
        return out
    if isinstance(expr, sp.Pow):
        b, e = expr.as_base_exp()
        return _sympy_to_pyomo(b, pvars) ** _sympy_to_pyomo(e, pvars)
    # Elementary
    if expr.func == sp.exp:
        return pyo.exp(_sympy_to_pyomo(expr.args[0], pvars))
    if expr.func == sp.sin:
        return pyo.sin(_sympy_to_pyomo(expr.args[0], pvars))
    if expr.func == sp.cos:
        return pyo.cos(_sympy_to_pyomo(expr.args[0], pvars))
    if expr.func == sp.Abs:
        a = _sympy_to_pyomo(expr.args[0], pvars)
        return pyo.sqrt(a * a)
    raise TypeError(f"Unsupported sympy node: {type(expr)} -> {expr}")


# --- Helpers to skip trivial equations/degenerate comps ---
def _is_zero_sympy(e):
    if isinstance(e, (int, float)) and e == 0:  # exact 0
        return True
    if isinstance(e, sp.Basic) and e.equals(0):  # provably 0
        return True
    return False

def _add_eq(clist: ConstraintList, sym_expr, pvars):
    if _is_zero_sympy(sym_expr):
        clist.add(Constraint.Feasible)   # skip True
    else:
        clist.add(_sympy_to_pyomo(sym_expr, pvars) == 0)


def build_pyomo_model_for_game(game: ParametricGame, theta_val: np.ndarray) -> ConcreteModel:
    m = ConcreteModel()

    n = game.total_dim()
    m.N = range(n)

    # decision vector z with bounds
    m.z = Var(m.N, domain=Reals)
    for i in m.N:
        lb = float(game.lower[i]) if np.isfinite(game.lower[i]) else None
        ub = float(game.upper[i]) if np.isfinite(game.upper[i]) else None
        m.z[i].setlb(lb); m.z[i].setub(ub)

    # residual F(z, theta)
    F = game.F_sym
    z_syms = list(game.z_sym)
    th_syms = list(game.theta_sym)
    pvars = {sym: m.z[i] for i, sym in enumerate(z_syms)}
    tpar  = {sym: float(theta_val[i]) for i, sym in enumerate(th_syms)}
    F_sub = F.subs(tpar)

    # containers
    m.sta = ConstraintList()
    m.eq_individual = ConstraintList()
    m.eq_shared = ConstraintList()

    nx_sum = sum(game.primal_dimensions)
    ng_sum = sum(game.equality_dimensions)
    nh_sum = sum(game.inequality_dimensions)

    row = 0
    # Stationarity ∂L/∂x == 0
    for i in range(nx_sum):
        _add_eq(m.sta, F_sub[i, 0], pvars)
        row += 1

    # Individual equalities g_i == 0
    for i in range(ng_sum):
        _add_eq(m.eq_individual, F_sub[row + i, 0], pvars)
    row += ng_sum

    # Individual complementarity: 0 ≤ μ ⟂ h(x,θ) ≥ 0
    mu_start = sum(game.primal_dimensions) + sum(game.equality_dimensions)
    def comp_ind_rule(_m, k):
        h_sym = F_sub[row + k, 0]
        if _is_zero_sympy(h_sym):
            return Complementarity.Skip  # skip 0 ⟂ 0
        h_expr = _sympy_to_pyomo(h_sym, pvars)
        mu_var = _m.z[mu_start + k]
        return complements(0 <= mu_var, h_expr >= 0)
    m.comp_ind = Complementarity(range(nh_sum), rule=comp_ind_rule)
    row += nh_sum

    # Shared equalities g~ == 0
    for i in range(game.shared_equality_dimension):
        _add_eq(m.eq_shared, F_sub[row + i, 0], pvars)
    row += game.shared_equality_dimension

    # Shared complementarity: 0 ≤ μ~ ⟂ h~(x,θ) ≥ 0
    mu_tilde_start = n - game.shared_inequality_dimension
    def comp_shared_rule(_m, k):
        h_sym = F_sub[row + k, 0]
        if _is_zero_sympy(h_sym):
            return Complementarity.Skip
        htilde_expr = _sympy_to_pyomo(h_sym, pvars)
        mu_tilde_var = _m.z[mu_tilde_start + k]
        return complements(0 <= mu_tilde_var, htilde_expr >= 0)
    m.comp_shared = Complementarity(range(game.shared_inequality_dimension), rule=comp_shared_rule)

    # dummy objective (required; not used by PATH for MCP)
    m.obj = Objective(expr=sum(m.z[i] ** 2 for i in m.N), sense=minimize)
    return m


def solve_with_path(
    game: ParametricGame,
    theta_val: np.ndarray,
    path_exe: Optional[str] = None,
    tee: bool = False,
):
    m = build_pyomo_model_for_game(game, theta_val)
    opt = SolverFactory('path', executable=path_exe) if path_exe else SolverFactory('path')
    res = opt.solve(m, tee=tee)
    z = np.array([value(m.z[i]) for i in m.N], dtype=float)
    status = f"{res.solver.status}/{res.solver.termination_condition}"
    info = {"solver_message": str(res.solver.message)}
    return z, status, info


def total_dim(game: ParametricGame) -> int:
    return game.total_dim()
