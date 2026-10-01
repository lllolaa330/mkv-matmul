import pytest
from dataclasses import replace
from pathlib import Path

from mkv_matmul.config import load_hardware, load_problem
from mkv_matmul.descriptor import encode_descriptor, decode_descriptor
from mkv_matmul.descriptor_layout import fits_unsigned, DESCRIPTOR_STRUCT
from mkv_matmul.types import Schedule

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
    
@pytest.fixture
def descriptor_context():
    root = Path(__file__).resolve().parents[1]

    problem = load_problem(
        str(root / "examples" / "q_proj_prefill.yaml")
    )
    original_hw = load_hardware(
        str(root / "examples" / "zcu104_16x16.yaml")
    )
    hw = replace(original_hw, weight_buffer_bytes=1024)
    schedule = Schedule(16, 16, 32)

    return problem, hw, schedule
  
def test_encode_descriptor_matches_expected_bytes(descriptor_context):
    problem, hw, schedule = descriptor_context

    data = encode_descriptor(problem, hw, schedule)

    expected = bytes.fromhex(
        "31 56 4b 4d 01 00 01 00 "
        "03 00 00 00 10 00 00 00 "
        "00 08 00 00 00 08 00 00 "
        "10 00 10 00 20 00 00 00 "
        "00 00 00 10 00 00 00 00 "
        "00 00 00 20 00 00 00 00 "
        "00 00 00 30 00 00 00 00 "
        "00 08 00 08 00 08 00 00"
    )

    assert DESCRIPTOR_STRUCT.size == 64
    assert len(data) == 64
    assert data == expected
    
def test_encode_descriptor_rejects_stride_overflow(descriptor_context):
    problem, hw, schedule = descriptor_context
    invalid_problem = replace(problem, output_stride=65536)

    with pytest.raises(
        ValueError,
        match="DESCRIPTOR_FIELD_OUT_OF_RANGE:output_stride",
    ):
        encode_descriptor(invalid_problem, hw, schedule)

def test_descriptor_round_trip(descriptor_context):
    problem, hw, schedule = descriptor_context

    encoded = encode_descriptor(problem, hw, schedule)
    decoded = decode_descriptor(encoded)

    assert decoded.problem == replace(
        problem,
        name="decoded_matmul",
    )
    assert decoded.schedule == schedule
    assert decoded.flags == 0b11

@pytest.mark.parametrize("length", [63, 65])
def test_decode_descriptor_rejects_wrong_length(length):
    with pytest.raises(ValueError, match="长度"):
        decode_descriptor(bytes(length))
        
@pytest.mark.parametrize(
    "offset, replacement, expected_message",
    [
        (0, b"\x00\x00\x00\x00", "magic"),
        (4, b"\x02\x00", "version"),
        (6, b"\x02", "opcode"),
        (7, b"\x01", "mode"),
        (8, b"\x04", "flags"),
        (30, b"\x01", "reserved"),
        (62, b"\x01", "reserved"),
        (12, b"\x00\x00\x00\x00", "M"),
        (24, b"\x00\x00", "tile_m"),
        (56, b"\x00\x00", "input_stride"),
    ],
)
def test_decode_descriptor_rejects_invalid_fields(
    descriptor_context,
    offset,
    replacement,
    expected_message,
):
    problem, hw, schedule = descriptor_context

    data = bytearray(encode_descriptor(problem, hw, schedule))

    data[offset:offset + len(replacement)] = replacement

    with pytest.raises(ValueError, match=expected_message):
        decode_descriptor(bytes(data))