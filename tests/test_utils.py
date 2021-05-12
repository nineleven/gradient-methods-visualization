import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.utils import CountCalls


def test_countcalls_loop() -> None:
    @CountCalls
    def f() -> float:
        return 2 + 2

    for i in range(10):
        assert f.n_calls == i
        f()
    assert f.n_calls == 10


def test_countcalls_args() -> None:
    @CountCalls
    def f(a: float, b: float=4) -> float:
        return a + b

    for i in range(10):
        assert f.n_calls == i
        if i % 2 == 0:
            f(i)
        else:
            f(i, b=i // 2)
    assert f.n_calls == 10
