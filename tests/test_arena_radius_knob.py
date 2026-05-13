import copy
import sys
import types
import unittest

import numpy as np

try:
    import torch  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    torch.Tensor = object
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

from config_rl import config_for_train
from core.env import Env


class ArenaRadiusKnobTests(unittest.TestCase):
    def _base_cfg(self):
        cfg = copy.deepcopy(config_for_train())
        dim = 2 * int(cfg["D"])
        cfg["dyn"]["Ad"] = np.eye(dim, dtype=np.float32)
        cfg["dyn"]["Bd"] = np.zeros((dim, int(cfg["D"])), dtype=np.float32)
        return cfg

    def test_disabled_knob_ignores_staged_update_validation(self):
        cfg = self._base_cfg()
        cfg["arena_radius_knob"]["enabled"] = False

        env = Env(cfg)

        self.assertEqual(env._scheduled_arena_radius(), cfg["arena"]["r"])

    def test_fixed_schedule_ignores_staged_update_validation(self):
        cfg = self._base_cfg()
        cfg["arena_radius_knob"]["enabled"] = True
        cfg["arena_radius_knob"]["schedule"] = "fixed"
        cfg["arena_radius_knob"]["final_radius_m"] = 42.0

        env = Env(cfg)

        self.assertEqual(env._scheduled_arena_radius(), 42.0)

    def test_staged_schedule_still_validates_update_counts(self):
        cfg = self._base_cfg()
        cfg["arena_radius_knob"]["enabled"] = True
        cfg["arena_radius_knob"]["schedule"] = "staged"
        cfg["arena_radius_knob"]["stages"] = [
            {"radius_m": 20.0, "updates": 1},
            {"radius_m": 40.0, "updates": 1},
        ]
        cfg["total_updates"] = 3

        env = Env(cfg)

        with self.assertRaisesRegex(ValueError, "update counts must sum"):
            env._scheduled_arena_radius()

    def test_stage_specific_attacker_shell_overrides_global_bounds(self):
        cfg = self._base_cfg()
        cfg["train_ic_mode"] = "random_shell"
        cfg["train_min_sep"] = 0.0
        cfg["total_updates"] = 2
        cfg["r_def_min_m"] = 0.0
        cfg["r_def_max_m"] = 0.0
        cfg["r_att_min_m"] = 6.0
        cfg["r_att_max"] = 0.95
        cfg["arena_radius_knob"]["enabled"] = True
        cfg["arena_radius_knob"]["schedule"] = "staged"
        cfg["arena_radius_knob"]["sample_mode"] = "fixed"
        cfg["arena_radius_knob"]["integer_only"] = False
        cfg["arena_radius_knob"]["stages"] = [
            {"radius_m": 20.0, "updates": 1, "r_att_min": 0.10, "r_att_max": 0.20},
            {"radius_m": 40.0, "updates": 1, "r_att_min_m": 30.0, "r_att_max_m": 32.0},
        ]

        env = Env(cfg)

        env.curriculum_progress = 0.0
        env.reset()
        _, _, pA_list, _ = env._unpack(env.state)
        r_stage0 = float(np.linalg.norm(pA_list[0] - env.center))
        self.assertGreaterEqual(r_stage0, 2.0 - 1e-6)
        self.assertLessEqual(r_stage0, 4.0 + 1e-6)

        env.curriculum_progress = 1.0
        env.reset()
        _, _, pA_list, _ = env._unpack(env.state)
        r_stage1 = float(np.linalg.norm(pA_list[0] - env.center))
        self.assertGreaterEqual(r_stage1, 30.0 - 1e-6)
        self.assertLessEqual(r_stage1, 32.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
