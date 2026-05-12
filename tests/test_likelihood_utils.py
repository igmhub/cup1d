import numpy as np
from cup1d.likelihood.likelihood import get_bin_coverage

def test_get_bin_coverage():
    xmin_o = np.array([0.0, 1.0, 2.0])
    xmax_o = np.array([1.0, 2.0, 3.0])
    xmin_n = np.array([0.5, 1.5])
    xmax_n = np.array([1.5, 2.5])
    
    cover = get_bin_coverage(xmin_o, xmax_o, xmin_n, xmax_n)
    
    # Expected coverage:
    # New bin 0 [0.5, 1.5] covers:
    # 0.5 of old bin 0 [0, 1]
    # 0.5 of old bin 1 [1, 2]
    # 0.0 of old bin 2 [2, 3]
    
    expected = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5]
    ])
    
    np.testing.assert_allclose(cover, expected)
