"""
tests/test_torch_vec_ekf.py

Regression tests that compare vectorized EKF behavior against scalar environments.
"""

import copy
import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    if not hasattr(torch, "Generator") or not hasattr(torch, "as_tensor"):
        torch = None

from config_rl import build_dyn, config_for_train

if torch is not None:
    from core.env import Env, TorchVecEnv
    from core.safety_filter import project_box_halfspace_torch, velocity_cbf_halfspace_torch
    from core.utils import set_seed
else:
    Env = None
    TorchVecEnv = None
    project_box_halfspace_torch = None
    velocity_cbf_halfspace_torch = None
    set_seed = None


def _deterministic_ekf_cfg():
    """Internal helper for deterministic ekf cfg."""
    cfg = copy.deepcopy(config_for_train())
    cfg["device"] = "cpu"
    cfg["use_kf"] = True
    cfg["estimator_kind"] = "ekf"
    cfg["reward_type"] = "zero_sum_kf"
    cfg["train_ic_mode"] = "fixed"
    cfg["x0_jitter"] = {"pos": 0.0, "vel": 0.0}
    cfg["num_envs"] = 1
    cfg["vec_backend"] = "torch"
    cfg["safety_filter"] = {
        "enabled": False,
        "kind": "velocity_cbf_qp",
        "alpha": 5.0,
        "vmax": 1.5,
    }
    cfg["ukf"]["every"] = 1
    cfg["ukf"]["sigma_az"] = 0.0
    cfg["ukf"]["sigma_el"] = 0.0
    cfg["ukf"]["init_mean_pos_std"] = 0.0
    cfg["ukf"]["init_mean_vel_std"] = 0.0
    cfg["ukf"]["init_mean_accel_std"] = 0.0
    cfg["ukf"]["action_access"] = "ground_truth"
    cfg["ukf"]["ekf_jacobian_mode"] = "exact"
    build_dyn(cfg)
    return cfg


class TorchVecEkfTests(unittest.TestCase):
    """Test case for matching torch-vectorized EKF behavior to scalar environment behavior."""

    @unittest.skipIf(torch is None, "torch is not installed in this environment")
    def test_torch_vec_ekf_matches_scalar_env_for_deterministic_step(self):
        """Verify that torch vec ekf matches scalar env for deterministic step behaves as expected."""
        cfg = _deterministic_ekf_cfg()
        set_seed(7)
        scalar_env = Env(copy.deepcopy(cfg))
        scalar_env.reset()

        set_seed(7)
        torch_env = TorchVecEnv(copy.deepcopy(cfg), num_envs=1, device="cpu")

        a1 = np.array([[0.12, -0.18, 0.05]], dtype=np.float32)
        a2 = np.array([[[-0.07, 0.09, -0.11]]], dtype=np.float32)

        obs_def_t0 = torch_env.obs_def[0].detach().cpu().numpy()
        obs_att_t0 = torch_env.obs_att[0].detach().cpu().numpy()
        obs_def_s0, obs_att_s0 = scalar_env.get_obs_pair()
        self.assertTrue(np.allclose(obs_def_t0, obs_def_s0, atol=1e-6))
        self.assertTrue(np.allclose(obs_att_t0, obs_att_s0, atol=1e-6))

        obs_def_t1, r1_t, r2_t, done_t, infos_t = torch_env.step(a1, a2, reward_mode="both")
        obs_def_s1, r1_s, r2_s, done_s, info_s = scalar_env.step(a1[0], a2[0], reward_mode="both")
        obs_att_t1 = torch_env.obs_att[0].detach().cpu().numpy()
        _, obs_att_s1 = scalar_env.get_obs_pair()

        self.assertTrue(np.allclose(obs_def_t1[0].detach().cpu().numpy(), obs_def_s1, atol=1e-5))
        self.assertTrue(np.allclose(obs_att_t1, obs_att_s1, atol=1e-5))
        self.assertAlmostEqual(float(r1_t[0].item()), float(r1_s), places=5)
        self.assertAlmostEqual(float(r2_t[0].item()), float(r2_s), places=5)
        self.assertEqual(bool(done_t[0].item()), bool(done_s))

        info_t = infos_t[0]
        for key in (
            "meas_innov_sq",
            "ukf_trPpos",
            "att_meas_innov_sq",
            "att_ukf_trPpos",
            "d2_belief_norm",
            "p2_est_err_norm",
            "d1_belief_norm",
            "p1_est_err_norm",
        ):
            self.assertIn(key, info_t)
            self.assertIn(key, info_s)
            self.assertAlmostEqual(float(info_t[key]), float(info_s[key]), places=5)

    @unittest.skipIf(torch is None, "torch is not installed in this environment")
    def test_torch_vec_velocity_cbf_uses_current_ltv_mats(self):
        """Verify that torch vec velocity cbf uses the current ltv matrices passed into the filter."""
        cfg = copy.deepcopy(config_for_train())
        cfg["device"] = "cpu"
        cfg["use_kf"] = False
        cfg["reward_type"] = "zero_sum"
        cfg["num_envs"] = 1
        cfg["vec_backend"] = "torch"
        cfg["train_ic_mode"] = "fixed"
        cfg["x0_jitter"] = {"pos": 0.0, "vel": 0.0}
        cfg["dynamics"] = "elliptic_ltv"
        cfg["dyn"]["type"] = "ltv"

        dim = 2 * int(cfg["D"])
        D = int(cfg["D"])
        dt = float(cfg["dt"])

        Ad_base = np.eye(dim, dtype=np.float32)
        Bd_base = np.zeros((dim, D), dtype=np.float32)

        Ad_cur = np.eye(dim, dtype=np.float32)
        Bd_cur = np.zeros((dim, D), dtype=np.float32)
        Bd_cur[D:, :] = np.eye(D, dtype=np.float32) * dt

        cfg["dyn"]["Ad"] = Ad_base
        cfg["dyn"]["Bd"] = Bd_base
        cfg["dyn"]["Ad_seq"] = np.stack([Ad_cur], axis=0)
        cfg["dyn"]["Bd_seq"] = np.stack([Bd_cur], axis=0)
        cfg["safety_filter"] = {
            "enabled": True,
            "kind": "velocity_cbf_qp",
            "alpha": 5.0,
            "vmax": 1.0,
        }

        env = TorchVecEnv(copy.deepcopy(cfg), num_envs=1, device="cpu")

        x = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=env.dtype, device=env.device)
        u_nom = torch.tensor([[0.5, 0.0, 0.0]], dtype=env.dtype, device=env.device)
        Ad_t = torch.as_tensor(Ad_cur, dtype=env.dtype, device=env.device)
        Bd_t = torch.as_tensor(Bd_cur, dtype=env.dtype, device=env.device)

        u_filtered = env._apply_velocity_cbf_filter_batch(x, u_nom, Ad_t=Ad_t, Bd_t=Bd_t)
        a, b = velocity_cbf_halfspace_torch(
            x,
            vmax=env.velocity_cbf_vmax,
            alpha=env.velocity_cbf_alpha,
            dyn_name=env._velocity_cbf_dyn_name,
            dt=env.dt,
            D=env.D,
            Ad_t=Ad_t,
            Bd_t=Bd_t,
            hcw_n=env._velocity_cbf_hcw_n,
        )
        expected = project_box_halfspace_torch(u_nom, env.u_lo_t, env.u_hi_t, a, b)

        self.assertTrue(torch.allclose(u_filtered, expected, atol=1e-6, rtol=1e-6))
        self.assertLessEqual(float(u_filtered[0, 0].item()), 1e-6)


if __name__ == "__main__":
    unittest.main()
