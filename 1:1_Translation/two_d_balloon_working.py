# =============================================
# file: two_d_balloon_working.py
# (Python analogue of examples/2D_Balloon_Working_OG.jl)
# =============================================
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple
from parametric_game import ParametricGame, solve_with_path
import sympy as sp


# --------- Simple planar double-integrator dynamics ---------
@dataclass
class PlanarDoubleIntegrator:
    x_bounds: Tuple[np.ndarray, np.ndarray]
    u_bounds: Tuple[np.ndarray, np.ndarray]

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # x = [px, py, vx, vy]; u = [ax, ay]
        dt = 1.0
        px, py, vx, vy = x
        ax, ay = u
        return np.array([px + dt * vx, py + dt * vy, vx + dt * ax, vy + dt * ay])

    @property
    def nx(self) -> int:
        return 4

    @property
    def nu(self) -> int:
        return 2


# --------- Helpers that mimic pack/unpack of trajectories ---------

def unpack_trajectory(flat: np.ndarray, nx_list: List[int], nu_list: List[int], H: int):
    """Return per-player dictionaries with xs, us lists of length H."""
    out = []
    offset = 0
    for nx, nu in zip(nx_list, nu_list):
        nblock = H * (nx + nu)
        block = flat[offset : offset + nblock]
        offset += nblock
        X = block[: H * nx].reshape(H, nx)
        U = block[H * nx :].reshape(H, nu)
        out.append({"xs": [X[t] for t in range(H)], "us": [U[t] for t in range(H)]})
    return out


def pack_trajectory(trajs: List[dict], nx_list: List[int], nu_list: List[int], H: int) -> np.ndarray:
    parts = []
    for p, (nx, nu) in enumerate(zip(nx_list, nu_list)):
        X = np.vstack(trajs[p]["xs"])  # H x nx
        U = np.vstack(trajs[p]["us"])  # H x nu
        parts.append(np.hstack([X.reshape(-1), U.reshape(-1)]))
    return np.concatenate(parts)


# --------- Build the game in symbolic form for the MCP layer ---------

def build_balloon_game(horizon: int = 10, balloon_pos=(0.0, 0.0)) -> ParametricGame:
    N = 4  # defender, attacker1, attacker2, balloon

    # per-player state/control sizes
    nx = [4, 4, 4, 4]
    nu = [2, 2, 2, 2]

    # Symbolic blocks helper: x_blocks[i] will be a sympy vector for player i's stacked (x,u) across horizon
    # For MCP, each player's primal is a flattened [x(1..H); u(1..H)]

    def stage_cost_factory():
        bx, by = balloon_pos

        def cost_i(x_blocks: List[sp.Matrix], theta: sp.Matrix) -> List[sp.Expr]:
            # We assume x_blocks[i] contains the whole trajectory. Here we only build a simple
            # surrogate cost using initial positions/velocities to keep this concise.
            def head_pos(block: sp.Matrix) -> Tuple[sp.Expr, sp.Expr]:
                # Take the first state's px, py in this simple surrogate
                return block[0, 0], block[1, 0]

            px1, py1 = head_pos(x_blocks[0])
            px2, py2 = head_pos(x_blocks[1])
            px3, py3 = head_pos(x_blocks[2])
            # balloon fixed at (bx, by)

            # squared distances
            d2_1 = (px1 - bx) ** 2 + (py1 - by) ** 2
            d2_2 = (px2 - bx) ** 2 + (py2 - by) ** 2
            d2_3 = (px3 - bx) ** 2 + (py3 - by) ** 2

            # defender tries to *maximize* distance of attackers to balloon and penalize its control (sign via MCP)
            J1 = -d2_2 - d2_3
            J2 = d2_2
            J3 = d2_3
            J4 = 0  # balloon is passive
            return [J1, J2, J3, J4]

        return cost_i

    cfun = stage_cost_factory()

    def objective_i(i: int) -> Callable[[List[sp.Matrix], sp.Matrix], sp.Expr]:
        def f_i(x_blocks: List[sp.Matrix], theta: sp.Matrix) -> sp.Expr:
            return cfun(x_blocks, theta)[i]
        return f_i

    objectives = [objective_i(i) for i in range(N)]

    # per-player individual constraints (dummy)
    equalities = [lambda x, th: sp.Matrix([0]) for _ in range(N)]
    inequalities = [lambda x, th: sp.Matrix([0]) for _ in range(N)]

    # shared constraints: simple initial-state equality (parameter is initial stacked state of all players at t=0)
    param_dim = sum(nx)

    def shared_eq(x_blocks: List[sp.Matrix], theta: sp.Matrix) -> sp.Matrix:
        # enforce first two entries of each player equal to theta slices (toy example)
        parts = []
        base = 0
        for i in range(N):
            parts.append(x_blocks[i][0, 0] - theta[base + 0, 0])
            parts.append(x_blocks[i][1, 0] - theta[base + 1, 0])
            base += nx[i]
        return sp.Matrix(parts)

    def shared_ineq(x_blocks: List[sp.Matrix], theta: sp.Matrix) -> sp.Matrix:
        return sp.Matrix([0])

    primals = [horizon * (nx[i] + nu[i]) for i in range(N)]

    game = ParametricGame(
        objectives=objectives,
        equality_constraints=equalities,
        inequality_constraints=inequalities,
        shared_equality_constraint=shared_eq,
        shared_inequality_constraint=shared_ineq,
        parameter_dimension=param_dim,
        primal_dimensions=primals,
        equality_dimensions=[1] * N,
        inequality_dimensions=[1] * N,
        shared_equality_dimension=sum([2 for _ in range(N)]),
        shared_inequality_dimension=1,
    )

    return game


def demo():
    game = build_balloon_game(horizon=10, balloon_pos=(0.0, 0.0))
    theta0 = np.zeros(game.parameter_dimension)
    z, status, info = solve_parametric_game(game, theta_val=theta0, verbose=True)
    print("status:", status, "info:", info)
    return z, status, info


from parametric_game import solve_with_path

if __name__ == "__main__":
    game = build_balloon_game(horizon=10, balloon_pos=(0.0, 0.0))
    theta0 = np.zeros(game.parameter_dimension)

    # Optional: confirm PATH availability/location
    from pyomo.opt import SolverFactory
    opt = SolverFactory('path', executable="/Users/gussantaella/Documents/UTAustin/Research/Code/Research_Repo/path_5/ampl/pathampl")
    print("Available?", opt.available(False))
    print("Executable:", opt.executable())

    # raise('Debug')

    # Set path_exe=".../path" if not on your $PATH
    from parametric_game import solve_with_path
    z, status, info = solve_with_path(game, theta0, path_exe="/Users/gussantaella/Documents/UTAustin/Research/Code/Research_Repo/path_5/ampl/pathampl", tee=True)
    print("STATUS:", status)
    print("INFO:", info)