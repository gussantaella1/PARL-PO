# balloon_2d_working_py.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple
from game_3dutils import di_discrete_AB, dims_from_D, rollout_linear, pack_trajectory, unpack_trajectory
from parametric_game import ParametricGameSpec, ParametricGameMCP

# ---------- Game assembly (mirrors Julia) ----------

def setup_trajectory_game_2d(balloon_pos=(0.0, 0.0),
                             attacker_speed=(1.0, 1.0),
                             defender_speed=(5.0, 10.0),
                             horizon=10, dt=0.1) -> Tuple[ParametricGameSpec, dict]:
    """
    Defender (player 1), attacker (player 2), and an optional extra attacker (player 3) / balloon (player 4) can be added later.
    Here: 1 v 1 (+ fixed balloon inside shared constraints).
    """
    D = 2
    nx, nu = dims_from_D(D)
    Ad, Bd = di_discrete_AB(dt, D=D)

    # PACKING: per-player τᵢ = [X(0..T-1), U(0..T-2)] => dim = T*nx + (T-1)*nu
    T = int(horizon)
    nprim = T*nx + (T-1)*nu

    # --- Costs f1,f2 (defender vs attacker-to-center) ---
    # Use the exact "guard_center" strategy like you added to game_costs.py.
    # Here I inline a Pyomo-expression version for self-contained 2D demo.
    def pos_of(tau_i, k):  # tau_i is flat packed
        X, U = unpack_trajectory(tau_i, nx, nu, T)
        return [X[k,0], X[k,1]]
    def vel_of(tau_i, k):
        X, U = unpack_trajectory(tau_i, nx, nu, T)
        return [X[k,2], X[k,3]]

    goal = np.asarray(balloon_pos, float)

    def f_defender(taus, theta, i):
        tau1, tau2 = taus[0], taus[1]
        J = 0.0
        eps = 1e-9
        P_def, eps_div = 1.0, 1e-2
        w_u1, w_v1 = 1e-2, 0.0
        w_center, w_term = 1.0, 8.0
        for k in range(T-1):
            p1 = pos_of(tau1, k); v1 = vel_of(tau1, k)
            p2 = pos_of(tau2, k)
            d12_sq = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
            # "reward" terms (negated for minimization):
            reward_prox = P_def / (eps_div + d12_sq)
            reward_push = ((p2[0]-goal[0])**2 + (p2[1]-goal[1])**2 + eps)**0.5
            J += -(reward_prox + reward_push)
            # effort on u1[k]
            _, U1 = unpack_trajectory(tau1, nx, nu, T)
            J += w_u1 * sum(U1[k,j]**2 for j in range(nu)) + w_v1 * sum(v1[j]**2 for j in range(D))
        # terminal push-out (negated so it maximizes distance-to-goal)
        p2T = pos_of(tau2, T-1)
        J += -w_term * (((p2T[0]-goal[0])**2 + (p2T[1]-goal[1])**2 + eps)**0.5)
        return J

    def f_attacker(taus, theta, i):
        tau1, tau2 = taus[0], taus[1]
        J = 0.0
        eps = 1e-9
        P_att, eps_div = 1.0, 1e-2
        w_u2, w_v2 = 1e-2, 0.0
        for k in range(T-1):
            p1 = pos_of(tau1, k); p2 = pos_of(tau2, k); v2 = vel_of(tau2, k)
            d12_sq = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
            # go to goal + repulsion from defender + effort
            J += ((p2[0]-goal[0])**2 + (p2[1]-goal[1])**2 + eps)**0.5
            J += P_att / (eps_div + d12_sq)
            _, U2 = unpack_trajectory(tau2, nx, nu, T)
            J += w_u2 * sum(U2[k,j]**2 for j in range(nu)) + w_v2 * sum(v2[j]**2 for j in range(D))
        # terminal: strong goal
        p2T = pos_of(tau2, T-1)
        J += 8.0 * (((p2T[0]-goal[0])**2 + (p2T[1]-goal[1])**2 + eps)**0.5)
        return J

    # Wrap as Pyomo-expression producing callables: f_i(τ,θ,i)
    def f_i_builder(i):
        def f(taulist, theta, _i=i):
            taus = [np.asarray(taulist)[sum(spec.primal_dims[:p]):sum(spec.primal_dims[:p+1])] for p in range(2)]
            if _i == 0:
                return f_defender(taus, theta, _i)
            else:
                return f_attacker(taus, theta, _i)
        return f

    # --- individual constraints: none (0-dim) ---
    gs = [lambda tau, th, i=i: [] for i in range(2)]
    hs = [lambda tau, th, i=i: [] for i in range(2)]

    # --- shared equalities: IC + linear dynamics (for both players) ---
    def shared_g(taulist, theta):
        # theta stacks both initial states: θ = [x1(0); x2(0)]
        taus = []
        ofs = 0
        for p in range(2):
            tau_p = taulist[ofs:ofs+nprim]; ofs += nprim
            taus.append(tau_p)
        g_list = []
        for p, tau_p in enumerate(taus):
            X, U = unpack_trajectory(tau_p, nx, nu, T)
            x0 = np.array(theta[p*nx:(p+1)*nx], float)
            g_list.extend(list(X[0] - x0))  # IC
            for k in range(T-1):
                g_list.extend(list(X[k+1] - (Ad @ X[k] + Bd @ U[k])))
        return g_list

    # --- shared inequalities: arena, speed, min separation, and defender cannot pop balloon ---
    def shared_h(taulist, theta):
        taus = []
        ofs = 0
        for p in range(2):
            tau_p = taulist[ofs:ofs+nprim]; ofs += nprim
            taus.append(tau_p)
        X1, U1 = unpack_trajectory(taus[0], nx, nu, T)
        X2, U2 = unpack_trajectory(taus[1], nx, nu, T)

        # Params
        arena_R = 10.0
        sep_min = 1.0
        r_balloon = 0.2

        H = []
        # Keep-in circle, speeds, separation, and defender cannot pop the balloon
        for k in range(T):
            p1 = X1[k,:2]; v1 = X1[k,2:]
            p2 = X2[k,:2]; v2 = X2[k,2:]

            # inside circle: R^2 - ||p||^2 >= 0
            H.append(arena_R**2 - (p1[0]**2 + p1[1]**2))
            H.append(arena_R**2 - (p2[0]**2 + p2[1]**2))
            # speed caps (optional) — comment out if using box bounds elsewhere
            vmax1 = defender_speed[0]
            vmax2 = attacker_speed[0]
            H.append(vmax1**2 - (v1[0]**2 + v1[1]**2))
            H.append(vmax2**2 - (v2[0]**2 + v2[1]**2))
            # separation
            d2 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            H.append(d2 - sep_min**2)
            # defender cannot touch balloon
            db2 = (p1[0]-goal[0])**2 + (p1[1]-goal[1])**2
            H.append(db2 - r_balloon**2)
        # input box bounds (optional): add if you do NOT want to rely on Var bounds
        for k in range(T-1):
            H.append((defender_speed[1]) - abs(U1[k,0])); H.append((defender_speed[1]) - abs(U1[k,1]))
            H.append((attacker_speed[1]) - abs(U2[k,0])); H.append((attacker_speed[1]) - abs(U2[k,1]))
        return H

    # Spec
    objectives = [f_i_builder(0), f_i_builder(1)]
    spec = ParametricGameSpec(
        objectives=objectives,
        indiv_equalities=gs,
        indiv_inequalities=hs,
        shared_equality=shared_g,
        shared_inequality=shared_h,
        parameter_dim=2*nx,
        primal_dims=[nprim, nprim],
        equality_dims=[0, 0],
        inequality_dims=[0, 0],
        shared_equality_dim=2*nx + 2*(T-1)*nx,
        shared_inequality_dim=T*(2 + 2 + 1 + 1) + (T-1)*4,  # rough count; not used explicitly by Pyomo
    )

    return spec, {"Ad": Ad, "Bd": Bd, "nx": nx, "nu": nu, "T": T, "D": D}

def solve_once(x0_def, x0_att, horizon=10, dt=0.1, path_exe="pathampl", tee=True):
    spec, info = setup_trajectory_game_2d(horizon=horizon, dt=dt)
    mcp = ParametricGameMCP(spec)
    theta = np.concatenate([x0_def, x0_att])
    mcp.set_theta(theta)

    # zero-input warm start
    nx, nu, T = info["nx"], info["nu"], info["T"]
    U0 = np.zeros((T-1, nu))
    X1 = rollout_linear(info["Ad"], info["Bd"], x0_def, U0)
    X2 = rollout_linear(info["Ad"], info["Bd"], x0_att, U0)
    tau1 = pack_trajectory(X1, U0)
    tau2 = pack_trajectory(X2, U0)
    mcp.warm_start_flat([tau1, tau2])

    mcp.solve_with_path(path_exe=path_exe, tee=tee)
    taus = mcp.extract_tau()

    # unpack for convenience
    X1_opt, U1_opt = unpack_trajectory(taus[0], nx, nu, T)
    X2_opt, U2_opt = unpack_trajectory(taus[1], nx, nu, T)
    return X1_opt, U1_opt, X2_opt, U2_opt
