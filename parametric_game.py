from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Sequence, Dict, Any

from pyomo.environ import (
    ConcreteModel, Var, Param, RangeSet, NonNegativeReals, Reals, Block, Constraint, value, SolverFactory
)
from pyomo.mpec import Complementarity, complements
from pyomo.core.expr.calculus.derivatives import differentiate, Modes


@dataclass
class ParametricGameSpec:
    objectives: List[Callable]                 # f_i(tau, theta, i) -> scalar Pyomo expr
    indiv_equalities: List[Callable]           # g_i(tau, theta, i) -> list[expr]
    indiv_inequalities: List[Callable]         # h_i(tau, theta, i) -> list[expr]
    shared_equality: Callable                  # g_tilde(tau, theta) -> list[expr]
    shared_inequality: Callable                # h_tilde(tau, theta) -> list[expr]
    parameter_dim: int
    primal_dims: Sequence[int]
    equality_dims: Sequence[int]
    inequality_dims: Sequence[int]
    shared_equality_dim: int
    shared_inequality_dim: int


class ParametricGame:
    def __init__(self, spec: ParametricGameSpec):
        self.spec = spec
        self._build_model()

    def _build_model(self):
        sp = self.spec
        N = len(sp.objectives)
        assert N == len(sp.primal_dims) == len(sp.equality_dims) == len(sp.inequality_dims)

        m = ConcreteModel()
        m.N = N
        m.P = RangeSet(0, N-1)

        # ---- θ as MUTABLE PARAM (not a variable) ----
        m.theta = Param(RangeSet(0, sp.parameter_dim-1), initialize=0.0, mutable=True)

        # one Block per player (attach Vars as real Pyomo components)
        m.player = Block(m.P)

        for i in m.P:
            nprim = int(sp.primal_dims[i])
            ngeq  = int(sp.equality_dims[i])
            nhiq  = int(sp.inequality_dims[i])

            b = m.player[i]
            b.Ix = RangeSet(0, nprim-1) if nprim > 0 else RangeSet(0, -1)
            b.x  = Var(b.Ix, domain=Reals, bounds=(-1e8, 1e8)) if nprim > 0 else None

            if ngeq > 0:
                b.Ig = RangeSet(0, ngeq-1)
                b.lam = Var(b.Ig, domain=Reals, bounds=(-1e12, 1e12))
            else:
                b.Ig = RangeSet(0, -1)
                b.lam = None

            if nhiq > 0:
                b.Ih = RangeSet(0, nhiq-1)
                b.mu  = Var(b.Ih, domain=NonNegativeReals)
            else:
                b.Ih = RangeSet(0, -1)
                b.mu = None

        # shared multipliers
        if sp.shared_equality_dim > 0:
            m.It = RangeSet(0, sp.shared_equality_dim-1)
            m.lam_t = Var(m.It, domain=Reals, bounds=(-1e12, 1e12))
        else:
            m.It = RangeSet(0, -1)
            m.lam_t = None

        if sp.shared_inequality_dim > 0:
            m.Iht = RangeSet(0, sp.shared_inequality_dim-1)
            m.mu_t  = Var(m.Iht, domain=NonNegativeReals)
        else:
            m.Iht = RangeSet(0, -1)
            m.mu_t = None

        # helpers to assemble tau/theta as flat lists of Vars/Params (Pyomo exprs)
        def tau_cat(_m):
            tau = []
            for i in _m.P:
                b = _m.player[i]
                if b.x is not None:
                    for k in b.Ix:
                        tau.append(b.x[k])
            return tau

        def theta_vec(_m):
            if sp.parameter_dim == 0:
                return []
            return [ _m.theta[i] for i in RangeSet(0, sp.parameter_dim-1) ]

        # short-hands to call user functions
        def f_i(_m, i):   return sp.objectives[i](tau_cat(_m), theta_vec(_m))
        def g_i(_m, i):   return sp.indiv_equalities[i](tau_cat(_m), theta_vec(_m)) if sp.equality_dims[i] > 0 else []
        def h_i(_m, i):   return sp.indiv_inequalities[i](tau_cat(_m), theta_vec(_m)) if sp.inequality_dims[i] > 0 else []
        def g_sh(_m):     return sp.shared_equality(tau_cat(_m), theta_vec(_m)) if sp.shared_equality_dim > 0 else []
        def h_sh(_m):     return sp.shared_inequality(tau_cat(_m), theta_vec(_m)) if sp.shared_inequality_dim > 0 else []

        # ---------------- Stationarity on indexed Block, using transfer ----------------
        m.st = Block(m.P)
        for i in m.P:
            b = m.player[i]
            # build on a temporary concrete block then transfer
            bi = Block(concrete=True)

            if b.x is not None:
                def _st_rule(_b, kk, ii=i):
                    xi = b.x[kk]
                    grad = differentiate(f_i(m, ii), wrt=xi, mode=Modes.reverse_symbolic)
                    if sp.equality_dims[ii] > 0:
                        Gi = g_i(m, ii)
                        for r in range(sp.equality_dims[ii]):
                            grad -= b.lam[r] * differentiate(Gi[r], wrt=xi, mode=Modes.reverse_symbolic)
                    if sp.inequality_dims[ii] > 0:
                        Hi = h_i(m, ii)
                        for r in range(sp.inequality_dims[ii]):
                            grad -= b.mu[r] * differentiate(Hi[r], wrt=xi, mode=Modes.reverse_symbolic)
                    if sp.shared_equality_dim > 0:
                        Gt = g_sh(m)
                        for r in range(sp.shared_equality_dim):
                            grad -= m.lam_t[r] * differentiate(Gt[r], wrt=xi, mode=Modes.reverse_symbolic)
                    if sp.shared_inequality_dim > 0:
                        Ht = h_sh(m)
                        for r in range(sp.shared_inequality_dim):
                            grad -= m.mu_t[r] * differentiate(Ht[r], wrt=xi, mode=Modes.reverse_symbolic)
                    return grad == 0

                bi.st = Constraint(b.Ix, rule=_st_rule)

            # <<< Option A here: transfer the temporary block into the indexed slot
            m.st[i].transfer_attributes_from(bi)

        # ---------------- per-player equalities g_i = 0 ----------------
        m.eq = Block(m.P)
        for i in m.P:
            if sp.equality_dims[i] > 0:
                def _geq_rule(_b, r, ii=i):
                    return g_i(m, ii)[r] == 0
                m.eq[i].g = Constraint(RangeSet(0, sp.equality_dims[i]-1), rule=_geq_rule)

        # ---------------- per-player complementarity 0 ≤ μ_i ⟂ h_i ≥ 0 ----------------
        m.comp = Block(m.P)
        for i in m.P:
            if sp.inequality_dims[i] > 0:
                def _comp_rule(_b, r, ii=i):
                    return complements(m.player[ii].mu[r] >= 0, h_i(m, ii)[r] >= 0)
                m.comp[i].c = Complementarity(RangeSet(0, sp.inequality_dims[i]-1), rule=_comp_rule)

        # ---------------- shared equalities g̃ = 0 ----------------
        if sp.shared_equality_dim > 0:
            def _gt_rule(_m, r):
                return g_sh(_m)[r] == 0
            m.eq_t = Constraint(RangeSet(0, sp.shared_equality_dim-1), rule=_gt_rule)

        # ---------------- shared complementarity 0 ≤ μ̃ ⟂ h̃ ≥ 0 ----------------
        if sp.shared_inequality_dim > 0:
            def _ht_rule(_m, r):
                return complements(m.mu_t[r] >= 0, h_sh(_m)[r] >= 0)
            m.comp_t = Complementarity(RangeSet(0, sp.shared_inequality_dim-1), rule=_ht_rule)

        self.m = m

    # ---------------- API (compatible with your caller) ----------------
    def set_theta(self, theta: Sequence[float]):
        assert len(theta) == self.spec.parameter_dim
        for i, v in enumerate(theta):
            self.m.theta[i] = float(v)   # Param is mutable

    def warm_start_flat(self, tau0_by_player: List[np.ndarray]):
        for i, tau in enumerate(tau0_by_player):
            b = self.m.player[i]
            if b.x is None:
                continue
            flat = np.asarray(tau, float).ravel()
            for k, val in zip(b.Ix, flat):
                b.x[k].set_value(float(val))

    def solve_with_path(self, path_exe="pathampl", tee=True, options=None):
            """
            Solve the MCP directly with PATH ("pathampl" or "path") instead of the ASL driver.
            """
            # Try the requested solver, then common fallbacks
            tried = []
            solver = None
            for name in [path_exe, "pathampl", "path"]:
                tried.append(name)
                opt = SolverFactory(name)
                if opt is not None and opt.available(False):
                    solver = opt
                    break
            if solver is None:
                raise RuntimeError(f"No executable found for PATH solver. Tried: {tried}")

            # Pass through user options (PATH ignores unknowns gracefully)
            for k, v in (options or {
                "proximal": 1e-2,
                "start": 1,
                "crash": 1,
                "major_iteration_limit": 20000,
            }).items():
                solver.options[str(k)] = v

            return solver.solve(
                self.m,
                tee=tee,
                keepfiles=True,
                symbolic_solver_labels=True,
                load_solutions=True,
            )

    def extract_tau(self) -> List[np.ndarray]:
        taus = []
        for i in self.m.P:
            b = self.m.player[i]
            if b.x is None:
                taus.append(np.zeros(0))
            else:
                vals = [value(b.x[k]) for k in b.Ix]
                taus.append(np.asarray(vals, float))
        return taus


# keep your existing import style in game_3dutils.py
def solve(game: ParametricGame, parameter_value: Sequence[float], initial_guess=None,
          path_exe="pathampl", options=None, tee=True) -> Dict[str, Any]:
    """
    Mirror the Julia-like API your framework expects.
    """
    game.set_theta(parameter_value)
    if initial_guess is not None:
        try:
            game.warm_start_flat(initial_guess)
        except Exception:
            pass
    game.solve_with_path(path_exe=path_exe, tee=tee, options=options)
    primals = game.extract_tau()
    return {
        "primals": primals,
        "variables": np.concatenate([p.ravel() for p in primals]) if primals else np.array([]),
        "status": "ok",
        "info": {},
    }
