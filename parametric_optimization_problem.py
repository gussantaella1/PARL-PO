# parametric_optimization_problem.py
from __future__ import annotations
import numpy as np
from pyomo.environ import *
from pyomo.mpec import Complementarity, complements
from pyomo.core.expr.calculus.derivatives import differentiate, Modes

class ParametricOptimizationProblem:
    """
    Mirror of the Julia single-player MCP wrapper:
      min_x f(x, θ)  s.t. g(x,θ)=0, h(x,θ) >= 0
    Expressed as a single MCP using KKT + comp(h, μ).
    """
    def __init__(self, f_builder, g_builder, h_builder,
                 n_prim: int, n_eq: int, n_ineq: int, p_dim: int = 1):
        self.f_builder = f_builder
        self.g_builder = g_builder
        self.h_builder = h_builder
        self.nx = n_prim
        self.ng = n_eq
        self.nh = n_ineq
        self.pd = p_dim
        self._build_model()

    def _build_model(self):
        m = ConcreteModel()
        m.Ix = RangeSet(0, self.nx-1)
        m.Ig = RangeSet(0, self.ng-1) if self.ng>0 else []
        m.Ih = RangeSet(0, self.nh-1) if self.nh>0 else []

        # Variables
        BIG = 1e8
        m.x = Var(m.Ix, bounds=(-BIG, BIG))
        if self.ng>0:
            m.lam = Var(m.Ig, bounds=(-BIG, BIG))     # equality multipliers (free)
        if self.nh>0:
            m.mu  = Var(m.Ih, domain=NonNegativeReals) # inequality multipliers

        # Parameter θ as a Param vector (mutable)
        m.theta = Param(RangeSet(0, self.pd-1), initialize=lambda _m,i: 0.0, mutable=True)

        # Symbolic helpers
        def vec_x(_m): return [_m.x[i] for i in _m.Ix]
        def vec_th(_m): return [_m.theta[i] for i in RangeSet(0, self.pd-1)] if self.pd>0 else []

        # Build f,g,h expressions
        def f_expr(_m):
            return self.f_builder(vec_x(_m), vec_th(_m))
        def g_expr_list(_m):
            if self.ng==0: return []
            return self.g_builder(vec_x(_m), vec_th(_m))
        def h_expr_list(_m):
            if self.nh==0: return []
            return self.h_builder(vec_x(_m), vec_th(_m))

        # Stationarity: ∇x L = ∇x f - Jg^T λ - Jh^T μ = 0
        def st_rule(_m, i):
            xi = _m.x[i]
            grad_f = differentiate(f_expr(_m), wrt=xi, mode=Modes.reverse_symbolic)
            JgTlam = 0.0
            if self.ng>0:
                for k in _m.Ig:
                    JgTlam += _m.lam[k] * differentiate(g_expr_list(_m)[k], wrt=xi, mode=Modes.reverse_symbolic)
            JhTmu = 0.0
            if self.nh>0:
                for k in _m.Ih:
                    JhTmu += _m.mu[k] * differentiate(h_expr_list(_m)[k], wrt=xi, mode=Modes.reverse_symbolic)
            return grad_f - JgTlam - JhTmu == 0
        m.st = Constraint(m.Ix, rule=st_rule)

        # Primal equalities
        if self.ng>0:
            def geq_rule(_m, k):
                return g_expr_list(_m)[k] == 0
            m.geq = Constraint(m.Ig, rule=geq_rule)

        # Complementarity μ ⟂ h >= 0
        if self.nh>0:
            def comp_rule(_m, k):
                return complements(_m.mu[k] >= 0, h_expr_list(_m)[k] >= 0)
            m.comp = Complementarity(m.Ih, rule=comp_rule)

        self.m = m

    def set_theta(self, theta):
        assert len(theta) == self.pd
        for i, v in enumerate(theta):
            self.m.theta[i] = float(v)

    def warm_start(self, x0=None, lam0=None, mu0=None):
        if x0 is not None:
            for i, v in enumerate(x0):
                self.m.x[i].value = float(v)
        if (lam0 is not None) and hasattr(self.m, 'lam'):
            for i, v in enumerate(lam0):
                self.m.lam[i].value = float(v)
        if (mu0 is not None) and hasattr(self.m, 'mu'):
            for i, v in enumerate(mu0):
                self.m.mu[i].value = float(v)

    def solve_with_path(self, path_exe="pathampl", tee=True, options=None):
        opt = SolverFactory("asl")
        opt.options["solver"] = path_exe
        for k, v in (options or {
            "proximal": 1e-2,
            "start": 1,
            "crash": 1,
            "major_iteration_limit": 20000,
        }).items():
            opt.options[str(k)] = v
        res = opt.solve(self.m, tee=tee, keepfiles=True, symbolic_solver_labels=True, load_solutions=True)
        return res
