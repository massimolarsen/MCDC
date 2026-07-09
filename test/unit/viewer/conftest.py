import pytest


@pytest.fixture(autouse=True)
def reset_simulation():
    from mcdc.object_.simulation import simulation

    simulation.__init__()
    yield
    simulation.__init__()
