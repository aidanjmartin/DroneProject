import pytest

from dronesim.params import load_gains, load_params


@pytest.fixture
def params():
    return load_params()


@pytest.fixture
def gains():
    return load_gains()
