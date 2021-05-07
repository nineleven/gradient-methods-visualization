import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pytest

from src.utils import CountCalls

def test_countcalls_loop():
    @CountCalls
    def f():
        return 2+2

    for i in range(10):
        assert f.n_calls == i
        f()
    assert f.n_calls == 10

def test_countcalls_args():
    @CountCalls
    def f(a, b=4):
        return a+b

    for i in range(10):
        assert f.n_calls == i
        if i % 2 == 0:
            f(i)
        else:
            f(i, b=i//2)
    assert f.n_calls == 10

