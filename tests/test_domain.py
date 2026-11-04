import numpy as np
import pytest

from roadlens.domain import decode_mask, encode_mask


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((3, 4), dtype=bool),
        np.ones((3, 4), dtype=bool),
        np.zeros((0, 4), dtype=bool),
        np.zeros((3, 0), dtype=bool),
        np.array([[False, True, False], [True, True, False]]),
    ],
)
def test_mask_rle_round_trip(mask):
    encoded = encode_mask(mask)
    assert np.array_equal(decode_mask(encoded), mask)


def test_mask_rle_rejects_incorrect_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        decode_mask({"width": 2, "height": 2, "counts": [3]})


@pytest.mark.parametrize("width,height", [(-2, -2), (-2, 0), (0, -2), (-1, 4), (4, -1)])
def test_mask_rle_rejects_negative_dimensions(width, height):
    with pytest.raises(ValueError, match="dimensions"):
        decode_mask({"width": width, "height": height, "counts": [width * height]})


def test_mask_rle_rejects_negative_runs_even_when_total_matches():
    with pytest.raises(ValueError, match="negative"):
        decode_mask({"width": 2, "height": 2, "counts": [3, -1, 2]})
