# neos_path_game_attitude.py
import os
import numpy as np
from pyomo.environ import *
from pyomo.mpec import Complementarity, complements
from pyomo.core.expr.calculus.derivatives import differentiate, Modes


# ----------------------- public helpers -----------------------
def derive_desired_dirs_from_plan(X_self, X_other, align="x"):
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
        d_seq.append(rel / (n + 1e-12) if n > 1e-12 else np.array([1.0,0.0,0.0]))
    return d_seq


def build_mcp_attitude_quat_single(
    T, dt, J_diag=(12.0,10.0,8.0), D_diag=(0.0,0.0,0.0),
    q0=(1,0,0,0), w0=(0,0,0),
    d_seq=None,
    w_track=50.0, w_w=1e-2, w_tau=1e-3, w_dw=1e-2,
    wmax=5.0, taumax=5.0,
):
    """
    One-agent nonlinear attitude planner for PATH.
    Variables per stage k: q[k,0..3], w[k,0..2]; controls: tau[k,0..2], k=0..T-2
    Dynamics:
      w_{k+1} = w_k + dt * J^{-1}(tau_k - w_k × (Jw_k) - D w_k)
      q_{k+1} = normalize((I + 0.5*dt*Omega(w_k)) q_k)
    Constraints:
      ||q_k||=1; |w_k|<=wmax; |tau_k|<=taumax
    Cost:
      Σ_k [ w_track*(1 - d_k^T * b_x(q_k)) + w_w*||w_k||^2 ] + Σ_k w_tau*||tau_k||^2 + Σ_k w_dw*||w_{k+1}-w_k||^2
    """
    import pyomo.environ as pyo
    m = pyo.ConcreteModel()

    # Sets
    m.K  = pyo.RangeSet(0, T-1)
    m.Ku = pyo.RangeSet(0, T-2)

    # Parameters
    d_seq = np.asarray(d_seq, float) if d_seq is not None else np.tile([1,0,0], (T,1))
    Jx,Jy,Jz = [float(J_diag[i]) for i in range(3)]
    Dx,Dy,Dz = [float(D_diag[i]) for i in range(3)]

    # Vars
    m.q   = pyo.Var(m.K, range(4), initialize=lambda m,k,i: 1.0 if (k==0 and i==0) else 0.0)
    m.w   = pyo.Var(m.K, range(3), initialize=0.0)
    m.tau = pyo.Var(m.Ku, range(3), bounds=(-taumax, taumax), initialize=0.0)

    # IC
    for i,val in enumerate(q0): m.q[0,i].fix(float(val))
    for i,val in enumerate(w0): m.w[0,i].fix(float(val))

    # Helpers
    def norm2_q(m,k):
        return sum(m.q[k,i]**2 for i in range(4))

    def bx_of_q(m,k):
        # body x row of world R(q)
        w,x,y,z = (m.q[k,0], m.q[k,1], m.q[k,2], m.q[k,3])
        # R[0,:] = [1-2(y^2+z^2), 2(xy-zw), 2(xz+yw)]
        return (
            1 - 2*(y*y + z*z),
            2*(x*y - z*w),
            2*(x*z + y*w),
        )

    # Unit norm at all k
    def unit_norm_rule(m,k):
        return norm2_q(m,k) == 1.0
    m.unit_norm = pyo.Constraint(m.K, rule=unit_norm_rule)

    # ω box bounds (path inequalities)
    # (Pyomo bounds could be used too; inequalities make it explicit)
    m.w_upper = pyo.Constraint(m.K, range(3), rule=lambda m,k,i:  m.w[k,i] <=  wmax)
    m.w_lower = pyo.Constraint(m.K, range(3), rule=lambda m,k,i: -m.w[k,i] <=  wmax)

    # Dynamics
    def dyn_w_rule(m,k,i):
        if k == T-1: return pyo.Constraint.Skip
        Jw = (
            Jx*m.w[k,0],
            Jy*m.w[k,1],
            Jz*m.w[k,2],
        )
        # w × (Jw)
        cx = m.w[k,1]*Jw[2] - m.w[k,2]*Jw[1]
        cy = m.w[k,2]*Jw[0] - m.w[k,0]*Jw[2]
        cz = m.w[k,0]*Jw[1] - m.w[k,1]*Jw[0]
        # D w
        Dw = (Dx*m.w[k,0], Dy*m.w[k,1], Dz*m.w[k,2])
        rhs = None
        if i == 0: rhs = m.w[k,0] + dt*( (m.tau[k,0] - cx - Dw[0]) / Jx )
        if i == 1: rhs = m.w[k,1] + dt*( (m.tau[k,1] - cy - Dw[1]) / Jy )
        if i == 2: rhs = m.w[k,2] + dt*( (m.tau[k,2] - cz - Dw[2]) / Jz )
        return m.w[k+1,i] == rhs
    m.dyn_w = pyo.Constraint(m.K, range(3), rule=dyn_w_rule)

    # q^+ ≈ normalize((I + 0.5 dt Ω(w)) q)
    # We enforce: q[k+1] == (I+0.5dtΩ)q / ||(I+0.5dtΩ)q||
    # Cross-multiplying by the norm introduces nonlinearity but PATH handles it.
    def dyn_q_rule(m,k,i):
        if k == T-1: return pyo.Constraint.Skip
        wx,wy,wz = m.w[k,0], m.w[k,1], m.w[k,2]
        # (I + 0.5dt Ω(w)) q
        # Ω(w)q in (w,x,y,z) order:
        # [ 0, -wx, -wy, -wz;
        #   wx, 0,  wz, -wy;
        #   wy,-wz, 0,  wx;
        #   wz, wy,-wx, 0 ] * q
        Omq = [
            -(wx*m.q[k,1] + wy*m.q[k,2] + wz*m.q[k,3]),
             wx*m.q[k,0] + wz*m.q[k,2] - wy*m.q[k,3],
             wy*m.q[k,0] - wz*m.q[k,1] + wx*m.q[k,3],
             wz*m.q[k,0] + wy*m.q[k,1] - wx*m.q[k,2],
        ]
        dq_i = m.q[k,i] + 0.5*dt*Omq[i]
        # norm of dq
        dq_norm = pyo.sqrt(sum( (m.q[k,j] + 0.5*dt*Omq[j])**2 for j in range(4) ))
        return m.q[k+1,i] * dq_norm == dq_i
    m.dyn_q = pyo.Constraint(m.K, range(4), rule=dyn_q_rule)

    # Objective
    def obj_rule(m):
        cost = 0.0
        for k in m.K:
            bx = bx_of_q(m,k)
            d  = d_seq[k]
            track = 1.0 - (d[0]*bx[0] + d[1]*bx[1] + d[2]*bx[2])   # in [0,2]
            cost += w_track*track + w_w*sum(m.w[k,i]**2 for i in range(3))
            if k < T-1:
                cost += w_tau*sum(m.tau[k,i]**2 for i in range(3))
        for k in range(T-1):
            cost += w_dw*sum( (m.w[k+1,i]-m.w[k,i])**2 for i in range(3))
        return cost
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return m
