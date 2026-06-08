"""
tests/test_velocity_cap.py

Regression tests for velocity-cap enforcement and rollout velocity reporting.
"""

import copy
import sys
import types
import unittest

import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
except ModuleNotFoundError:
    matplotlib = None

try:
    import torch  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.Generator = type("Generator", (), {})
    torch.dtype = type("dtype", (), {})
    torch.device = type("device", (), {})
    torch.float32 = object()
    torch.int64 = object()
    torch.bool = object()
    torch.manual_seed = lambda seed: None
    torch.clamp = lambda x, *args, **kwargs: x
    torch.log1p = lambda x: x
    torch.tanh = lambda x: x

    distributions = types.ModuleType("torch.distributions")
    distributions.Normal = type("Normal", (), {})
    torch.distributions = distributions

    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    functional.softplus = lambda x: x
    nn.functional = functional
    torch.nn = nn

    sys.modules["torch"] = torch
    sys.modules["torch.distributions"] = distributions
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional

ukf_estimator = types.ModuleType("ukf_estimator")
ukf_estimator.KF_CV = type("KF_CV", (), {})
ukf_estimator._body_bearing_from_world = lambda *args, **kwargs: None
ukf_estimator._azel_from_body_vec = lambda *args, **kwargs: None
sys.modules.setdefault("ukf_estimator", ukf_estimator)

from config_rl import config_for_train as config_for_train_single
from core.env import Env as SingleEnv
from paper_baseline_runner import _build_step_plant_single

if matplotlib is not None:
    from game_viz import plot_rollout_velocity
else:
    plot_rollout_velocity = None


def _base_cfg(factory):
    """Internal helper for base cfg."""
    cfg = copy.deepcopy(factory())
    dim = 2 * int(cfg["D"])
    cfg["dyn"]["Ad"] = np.eye(dim, dtype=np.float32)
    cfg["dyn"]["Bd"] = np.zeros((dim, int(cfg["D"])), dtype=np.float32)
    cfg["train_ic_mode"] = "fixed"
    cfg["x0_jitter"] = {"pos": 0.0, "vel": 0.0}
    cfg["safety_filter"] = {
        "enabled": True,
        "kind": "velocity_cbf_qp",
        "alpha": 5.0,
        "vmax": 2.5,
    }
    cfg["x0"] = np.array(
        [
            [1.0, 2.0, 3.0, 3.0, 4.0, 0.0],
            [-1.0, -2.0, -3.0, 0.0, -6.0, -8.0],
        ],
        dtype=float,
    )
    return cfg


class VelocityCapTests(unittest.TestCase):
    """Test case for velocity-cap safety behavior and rollout diagnostics."""
    def _assert_no_hard_clip_after_reset_and_step(self, env_cls, cfg):
        """Internal helper for assert no hard clip after reset and step."""
        env = env_cls(cfg)
        env.reset()

        state = np.asarray(env.state, dtype=float)
        D = int(cfg["D"])
        vmax = float(cfg["safety_filter"]["vmax"])

        self.assertGreater(np.linalg.norm(state[D:2 * D]), vmax + 1e-6)
        self.assertGreater(np.linalg.norm(state[3 * D:4 * D]), vmax + 1e-6)

        env.step(np.zeros(D, dtype=float), np.zeros((1, D), dtype=float))
        state = np.asarray(env.state, dtype=float)

        self.assertGreater(np.linalg.norm(state[D:2 * D]), vmax + 1e-6)
        self.assertGreater(np.linalg.norm(state[3 * D:4 * D]), vmax + 1e-6)

    def test_single_env_no_longer_hard_clips_velocity_state(self):
        """Verify that single env no longer hard clips velocity state behaves as expected."""
        self._assert_no_hard_clip_after_reset_and_step(SingleEnv, _base_cfg(config_for_train_single))

    def test_rollout_step_helper_no_longer_hard_clips_velocity_state(self):
        """Verify that rollout step helper no longer hard clips velocity state behaves as expected."""
        cfg = _base_cfg(config_for_train_single)
        step_plant_single, _, _ = _build_step_plant_single(cfg, steps=2, D=int(cfg["D"]))
        D = int(cfg["D"])
        vmax = float(cfg["safety_filter"]["vmax"])

        x_next = step_plant_single(np.asarray(cfg["x0"][1], dtype=np.float32), np.zeros(D, dtype=np.float32), 0)

        self.assertGreater(np.linalg.norm(x_next[D:2 * D]), vmax + 1e-6)

    def test_single_env_velocity_cbf_filter_blocks_outward_accel(self):
        """Verify that single env velocity cbf filter blocks outward accel behaves as expected."""
        cfg = _base_cfg(config_for_train_single)
        cfg["safety_filter"]["vmax"] = 1.0
        env = SingleEnv(cfg)

        p = np.zeros((int(cfg["D"]),), dtype=float)
        v = np.array([1.0, 0.0, 0.0], dtype=float)

        outward = env._apply_velocity_cbf_filter(p, v, np.array([env.u_hi, 0.0, 0.0], dtype=float))
        inward = env._apply_velocity_cbf_filter(p, v, np.array([-0.5 * env.u_hi, 0.0, 0.0], dtype=float))

        self.assertLessEqual(outward[0], 1e-6)
        self.assertTrue(np.allclose(inward, np.array([-0.5 * env.u_hi, 0.0, 0.0], dtype=float), atol=1e-6))

    @unittest.skipIf(plot_rollout_velocity is None, "matplotlib is not installed in this environment")
    def test_plot_rollout_velocity_reads_state_history(self):
        """Verify that plot rollout velocity reads state history behaves as expected."""
        D = 3
        vmax = 2.5
        states = np.array(
            [
                [0.0, 0.0, 0.0, 1.5, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, -3.0, -4.0],
                [0.1, 0.2, 0.0, 1.5, 2.0, 0.0, 0.9, -0.3, 0.0, 0.0, -3.0, -4.0],
            ],
            dtype=float,
        )

        fig, ax = plot_rollout_velocity(
            {"states": states},
            cfg={"D": D, "num_attackers": 1, "dt": 0.1, "vmax": vmax},
            agent=2,
            show=False,
        )
        try:
            plotted_speed = np.asarray(ax.lines[-1].get_ydata(), dtype=float)
            expected_speed = np.linalg.norm(states[:, 3 * D : 4 * D], axis=1)
            self.assertTrue(np.allclose(plotted_speed, expected_speed))
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
