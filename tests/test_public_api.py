import importlib.util

import pytest

pytest.importorskip("torch")
pytest.importorskip("gpytorch")


def test_public_import_surface():
    from lcbo_opt import LCBO, make_benchmark_problem

    assert LCBO.__name__ == "LCBO"
    assert callable(make_benchmark_problem)


def test_gymnasium_is_not_required_for_package_import():
    import lcbo_opt

    assert lcbo_opt.LCBO.__name__ == "LCBO"
    assert importlib.util.find_spec("lcbo_opt") is not None
