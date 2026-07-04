"""Tests that the reproduction shows the double-descent signature. Run: pytest -q"""
import numpy as np
from ddlab.experiment import double_descent_curve, make_data

def test_data_shapes():
    X, y, w = make_data(n=30, d=8, noise=0.3)
    assert X.shape == (30, 8) and y.shape == (30,)

def test_curve_has_peak_near_interpolation_threshold():
    ratios, errs = double_descent_curve(noise=0.5, ridge=1e-6, repeats=10)
    peak = int(np.argmax(errs))
    # the spike should sit near P/n = 1 (between 0.8 and 1.3)
    assert 0.8 <= ratios[peak] <= 1.3

def test_second_descent_below_peak():
    ratios, errs = double_descent_curve(noise=0.5, ridge=1e-6, repeats=10)
    peak = int(np.argmax(errs))
    # error in the heavily over-parameterized regime is far below the peak
    assert errs[-1] < errs[peak] / 5

def test_large_ridge_suppresses_peak():
    _, sharp = double_descent_curve(noise=0.5, ridge=1e-6, repeats=10)
    _, smooth = double_descent_curve(noise=0.5, ridge=1.0, repeats=10)
    assert max(smooth) < max(sharp)        # regularization removes the spike
