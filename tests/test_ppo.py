import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ppo import compute_gae, RunningMeanStd


def test_gae_hand_example():
    # T=3, one env. gamma=0.5, lam=0.5. rewards 1,1,1; values 0.5,1,2; last_value 4; no dones.
    # d2 = 1+0.5*4-2 = 1 ; d1 = 1+0.5*2-1 = 1 ; d0 = 1+0.5*1-0.5 = 1
    # A2 = 1 ; A1 = 1+0.25*1 = 1.25 ; A0 = 1+0.25*1.25 = 1.3125
    adv = compute_gae(np.ones((3, 1)), np.array([[.5], [1.], [2.]]), np.zeros((3, 1)), np.array([4.]), .5, .5)
    np.testing.assert_allclose(adv[:, 0], [1.3125, 1.25, 1.0])


def test_gae_done_cuts_bootstrap():
    # done after step 1: step 1 must not see V[2] or the advantage of step 2
    v = np.array([[1.], [2.], [3.]]); r = np.ones((3, 1)); d = np.array([[0.], [1.], [0.]])
    adv = compute_gae(r, v, d, np.array([10.]), .9, .9)
    assert np.isclose(adv[1, 0], 1 - 2)            # delta1 = r - V1, nothing after
    assert np.isclose(adv[0, 0], (1 + .9 * 2 - 1) + .9 * .9 * adv[1, 0])


def test_gae_lambda1_is_mc_return_minus_value():
    r = np.array([[1.], [2.], [3.]]); v = np.array([[.3], [.2], [.1]])
    adv = compute_gae(r, v, np.zeros((3, 1)), np.array([5.]), .9, 1.0)
    G = 1 + .9 * 2 + .81 * 3 + .729 * 5
    assert np.isclose(adv[0, 0], G - .3)


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, (1000, 5))
    rms = RunningMeanStd((5,))
    for c in np.array_split(x, 7):  # uneven-ish batches
        rms.update(c)
    np.testing.assert_allclose(rms.mean, x.mean(0), atol=1e-3)
    np.testing.assert_allclose(rms.var, x.var(0), rtol=1e-3)
