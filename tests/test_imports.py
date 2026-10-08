"""Package and submodule import smoke tests."""


def test_top_level_import():
    import robustcomm

    assert robustcomm.__version__


def test_subpackages_import():
    import robustcomm.aggregators  # noqa: F401
    import robustcomm.attacks  # noqa: F401
    import robustcomm.baselines  # noqa: F401
    import robustcomm.data  # noqa: F401
    import robustcomm.envs  # noqa: F401
    import robustcomm.experiments  # noqa: F401
    import robustcomm.metrics  # noqa: F401
    import robustcomm.policy  # noqa: F401
    import robustcomm.theory  # noqa: F401
    import robustcomm.training  # noqa: F401
    import robustcomm.utils  # noqa: F401


def test_public_api():
    from robustcomm import AGGREGATORS, build_aggregator, bounds

    assert set(AGGREGATORS) == {"krum", "median", "trmean", "mean"}
    assert callable(build_aggregator)
    assert hasattr(bounds, "krum_deviation_factor")
