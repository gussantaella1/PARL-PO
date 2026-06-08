"""
tests/test_intercept_heuristic.py

Regression tests for intercept-prior geometry and blend scheduling.
"""

import unittest

import numpy as np

from core.intercept_heuristic import (
    annealed_prior_blend,
    blended_intercept_target_np,
    clamp_intercept_mix,
    intercept_direction_prior_np,
)


class InterceptHeuristicTests(unittest.TestCase):
    """Test case for the intercept heuristic helpers."""
    def test_mix_zero_falls_back_to_current_attacker_position(self):
        """Verify that mix zero falls back to current attacker position behaves as expected."""
        dt = 0.1
        D = 3
        Ad = np.block(
            [
                [np.eye(D, dtype=np.float32), dt * np.eye(D, dtype=np.float32)],
                [np.zeros((D, D), dtype=np.float32), np.eye(D, dtype=np.float32)],
            ]
        )
        p_att = np.array([10.0, -3.0, 1.0], dtype=np.float32)
        v_att = np.array([2.0, 0.5, -1.0], dtype=np.float32)

        target = blended_intercept_target_np(
            Ad,
            p_att,
            v_att,
            lookahead_steps=10.0,
            mix=0.0,
        )

        self.assertTrue(np.allclose(target, p_att))

    def test_full_mix_uses_coasting_intercept_prediction(self):
        """Verify that full mix uses coasting intercept prediction behaves as expected."""
        dt = 0.1
        D = 3
        Ad = np.block(
            [
                [np.eye(D, dtype=np.float32), dt * np.eye(D, dtype=np.float32)],
                [np.zeros((D, D), dtype=np.float32), np.eye(D, dtype=np.float32)],
            ]
        )
        p_att = np.array([10.0, -3.0, 1.0], dtype=np.float32)
        v_att = np.array([2.0, 0.5, -1.0], dtype=np.float32)

        target = blended_intercept_target_np(
            Ad,
            p_att,
            v_att,
            lookahead_steps=10.0,
            mix=1.0,
        )

        expected = p_att + (10.0 * dt) * v_att
        self.assertTrue(np.allclose(target, expected, atol=1e-6))

    def test_mix_is_clamped_into_unit_interval(self):
        """Verify that mix is clamped into unit interval behaves as expected."""
        self.assertEqual(clamp_intercept_mix(-1.0), 0.0)
        self.assertEqual(clamp_intercept_mix(0.25), 0.25)
        self.assertEqual(clamp_intercept_mix(3.0), 1.0)

    def test_prior_blend_anneals_linearly_and_clamps(self):
        """Verify that prior blend anneals linearly and clamps behaves as expected."""
        self.assertAlmostEqual(
            annealed_prior_blend(
                0.0,
                start_blend=0.6,
                end_blend=0.0,
                anneal_fraction=1.0,
            ),
            0.6,
        )
        self.assertAlmostEqual(
            annealed_prior_blend(
                0.5,
                start_blend=0.6,
                end_blend=0.0,
                anneal_fraction=1.0,
            ),
            0.3,
        )
        self.assertAlmostEqual(
            annealed_prior_blend(
                0.5,
                start_blend=0.6,
                end_blend=0.0,
                anneal_fraction=0.25,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            annealed_prior_blend(
                0.5,
                start_blend=0.6,
                end_blend=0.2,
                anneal_fraction=0.0,
            ),
            0.2,
        )

    def test_intercept_direction_prior_points_to_blended_target(self):
        """Verify that intercept direction prior points to blended target behaves as expected."""
        dt = 0.1
        D = 3
        Ad = np.block(
            [
                [np.eye(D, dtype=np.float32), dt * np.eye(D, dtype=np.float32)],
                [np.zeros((D, D), dtype=np.float32), np.eye(D, dtype=np.float32)],
            ]
        )
        p_def = np.zeros(D, dtype=np.float32)
        p_att = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        v_att = np.array([2.0, 0.0, 0.0], dtype=np.float32)

        prior = intercept_direction_prior_np(
            Ad,
            p_def,
            p_att,
            v_att,
            lookahead_steps=10.0,
            mix=1.0,
            gain=2.0,
        )

        self.assertTrue(np.allclose(prior, np.array([2.0, 0.0, 0.0], dtype=np.float32), atol=1e-6))

    def test_intercept_direction_prior_is_zero_at_target(self):
        """Verify that intercept direction prior is zero at target behaves as expected."""
        dt = 0.1
        D = 3
        Ad = np.block(
            [
                [np.eye(D, dtype=np.float32), dt * np.eye(D, dtype=np.float32)],
                [np.zeros((D, D), dtype=np.float32), np.eye(D, dtype=np.float32)],
            ]
        )
        p_def = np.array([3.0, -1.0, 0.5], dtype=np.float32)
        p_att = p_def.copy()
        v_att = np.zeros(D, dtype=np.float32)

        prior = intercept_direction_prior_np(
            Ad,
            p_def,
            p_att,
            v_att,
            lookahead_steps=5.0,
            mix=0.5,
            gain=2.0,
        )

        self.assertTrue(np.allclose(prior, np.zeros(D, dtype=np.float32), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
