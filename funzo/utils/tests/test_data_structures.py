
import pytest
import h5py
import os

import numpy as np

from numpy.testing import assert_equal

from funzo.utils.data_structures import Trace


def test_trace_init():
    t = Trace()
    assert len(t['step']) == 0


def test_Trace_record():
    t = Trace()
    t.record(1, [1.0, 2.0], [1.0, 2.0], 0.7, np.random.rand(5), 0.6)
    t.record(2, [15.0, 2.0], [5.0, 2.0], 0.8, np.random.rand(5), 0.6)
    t.record(3, [4.0, 2.0], [1.0, 6.0], 0.8, np.random.rand(5), 0.6)
    assert len(t['step']) == 3
    assert len(t['Q_r']) == 3
    assert len(t['log_p']) == 3
    assert len(t['sample']) == 3
    assert len(t['a_ratio']) == 3

    # check assertions in the code due to invalid mix of values
    with pytest.raises(ValueError):
        t.record(1, [1.0, 2.0, 3.0], [1.0, 2.0], 0.7, np.random.rand(5), 0.6)
    with pytest.raises(ValueError):
        t.record(-1, [1.0, 2.0], [1.0, 2.0], 0.7, np.random.rand(5), 0.6)


def test_trace_save():
    t = Trace()

    np.random.seed(42)

    qq = np.random.rand(5)

    t.record(1, [1.0, 2.0], [1.0, 2.0], 0.7, qq, 0.6)
    fname = t.save('trace')

    f = h5py.File(fname, 'r')
    for k in t.vars:
        assert k in f

    assert_equal(f['r'][0], [1.0, 2.0])
    assert_equal(f['sample'][0], [1.0, 2.0])
    assert_equal(f['Q_r'][0], qq)

    os.remove(fname)


def test_trace_getitem():
    t = Trace()
    with pytest.raises(ValueError):
        t['V']
