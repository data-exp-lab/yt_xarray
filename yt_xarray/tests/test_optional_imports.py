import pytest

from yt_xarray.utilities._utilities import _import_optional_dep


def test_missing_import():
    with pytest.raises(ImportError, match="This functionality requires not_a_module"):
        _ = _import_optional_dep("not_a_module")

    with pytest.raises(ImportError, match="Nooooooope"):
        _ = _import_optional_dep("not_a_module", custom_message="Nooooooope")


def test_optional_import():
    # just make sure it returns the module as expected
    yt = _import_optional_dep("yt")
    assert hasattr(yt, "load_sample")
