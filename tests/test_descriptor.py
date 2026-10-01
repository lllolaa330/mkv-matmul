import pytest

from mkv_matmul.descriptor import fits_unsigned

@pytest.mark.parametrize(
    "value, bits, expected",
    [
        (0, 16, True),
        (65535, 16, True),
        (65536, 16, False),
        (-1, 16, False),
        (True, 16, False),
        (16.0, 16, False),
        (2**32 - 1, 32, True),
        (2**32, 32, False),
    ],
)
def test_fits_unsigned(value, bits, expected):
    assert fits_unsigned(value, bits) is expected