# =============================================
# file: assignment5/parametric_optimization_problem.py
# =============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import sympy as sp


@dataclass
class ParametricOptimizationProblem:
    """Python analogue of Julia's ParametricOptimizationProblem.

    All callables should accept (x, theta) and return sympy expressions/vectors
    when called with sympy Symbols; they may also accept numpy arrays if you
    only need numeric evaluation. This mirrors the Julia structure in
    parametric_optimization_problem.jl.
    """

    objective: Callable[[sp.Matrix, sp.Matrix], sp.Expr]
    equality_constraint: Callable[[sp.Matrix, sp.Matrix], sp.Matrix]
    inequality_constraint: Callable[[sp.Matrix, sp.Matrix], sp.Matrix]

    parameter_dimension: int
    primal_dimension: int
    equality_dimension: int
    inequality_dimension: int

    # derived symbolic containers (built in __post_init__)
    z_sym: Optional[sp.Matrix] = None
    theta_sym: Optional[sp.Matrix] = None
    F_sym: Optional[sp.Matrix] = None
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        total = self.primal_dimension + self.equality_dimension + self.inequality_dimension

        # decision blocks z = [x; lambda; mu]
        z = sp.Matrix(sp.symbols(f"z0:{total}"))  # column vector
        self.z_sym = z

        x = z[: self.primal_dimension, :]
        lam = z[self.primal_dimension : self.primal_dimension + self.equality_dimension, :]
        mu = z[self.primal_dimension + self.equality_dimension :, :]

        # parameters theta
        theta = sp.Matrix(sp.symbols(f"theta0:{self.parameter_dimension}"))
        self.theta_sym = theta

        # objective and constraints (symbolic)
        f = self.objective(x, theta)
        g = self.equality_constraint(x, theta)
        h = self.inequality_constraint(x, theta)

        if not isinstance(g, sp.Matrix):
            g = sp.Matrix(g)
        if not isinstance(h, sp.Matrix):
            h = sp.Matrix(h)

        # Lagrangian L = f - lam^T g - mu^T h (sign convention matches KKT stationarity)
        L = f - (lam.T * g)[0] - (mu.T * h)[0]

        # F = [grad_x L; g; h]
        dLdx = sp.Matrix([sp.diff(L, x[i, 0]) for i in range(self.primal_dimension)])
        F = sp.Matrix.vstack(dLdx, g, h)
        self.F_sym = F

        # bounds: x free; lam free (equalities); mu >= 0 (inequalities)
        lb = np.concatenate(
            [
                -np.inf * np.ones(self.primal_dimension),
                -np.inf * np.ones(self.equality_dimension),
                np.zeros(self.inequality_dimension),
            ]
        )
        ub = np.inf * np.ones(total)
        self.lower = lb
        self.upper = ub

    def total_dim(self) -> int:
        return self.primal_dimension + self.equality_dimension + self.inequality_dimension


# ---------- MCP Solve (placeholder) ----------
# This stub shows how to hook into Pyomo + PATH if desired.
# You can swap in your existing PATH/pyomo stack.

def solve_parametric_optimization(
    problem: ParametricOptimizationProblem,
    theta_val: Optional[np.ndarray] = None,
    initial_guess: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, str, dict]:
    """Solve using a simple projected Newton fallback or delegate to Pyomo+PATH.

    Returns: primals (x), full z, status, info
    """
    n = problem.total_dim()
    if theta_val is None:
        theta_val = np.zeros(problem.parameter_dimension)
    if initial_guess is None:
        z = np.zeros(n)
    else:
        z = np.array(initial_guess, dtype=float).copy()

    # Simple fixed-point / projected gradient fallback for small problems.
    # For production, replace with Pyomo+PATH complementarity solve.
    xdim = problem.primal_dimension
    gdim = problem.equality_dimension
    hdim = problem.inequality_dimension

    x_idx = slice(0, xdim)
    lam_idx = slice(xdim, xdim + gdim)
    mu_idx = slice(xdim + gdim, xdim + gdim + hdim)

    F = problem.F_sym
    z_syms = list(problem.z_sym)
    th_syms = list(problem.theta_sym)

    F_num = sp.lambdify((z_syms, th_syms), F, "numpy")

    def proj(zvec: np.ndarray) -> np.ndarray:
        zvec = np.minimum(np.maximum(zvec, problem.lower), problem.upper)
        return zvec

    iters = 0
    status = "Not Converged"
    for iters in range(500):
        Fval = np.array(F_num(z.tolist(), theta_val.tolist()), dtype=float).reshape(-1)
        normF = np.linalg.norm(Fval)
        if verbose and iters % 25 == 0:
            print(f"iter {iters}: ||F|| = {normF:.3e}")
        if normF < 1e-6:
            status = "Solved"
            break
        # very basic damped step
        z = proj(z - 1e-2 * Fval)
        # keep mu >= 0 explicitly (already enforced by bounds)
        z[mu_idx] = np.maximum(z[mu_idx], 0.0)

    primals = z[x_idx]
    info = {"iterations": iters, "residual": float(np.linalg.norm(Fval))}
    return primals, z, status, info


def total_dim(problem: ParametricOptimizationProblem) -> int:
    return problem.total_dim()
