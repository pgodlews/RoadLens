import json

import pytest

from roadlens.report import generate_report, parse_byte_range


def test_parse_byte_range():
    assert parse_byte_range(None, 100) is None
    assert parse_byte_range("bytes=10-19", 100) == (10, 19)
    assert parse_byte_range("bytes=90-", 100) == (90, 99)
    assert parse_byte_range("bytes=-10", 100) == (90, 99)
    assert parse_byte_range("bytes=90-200", 100) == (90, 99)


@pytest.mark.parametrize("value", ["bytes=100-101", "bytes=20-10", "items=1-2"])
def test_parse_byte_range_rejects_invalid_input(value):
    with pytest.raises(ValueError):
        parse_byte_range(value, 100)


def test_generate_report_embeds_observations_and_tile_url(tmp_path):
    observation = {
        "object_id": 1,
        "prompt": "pothole",
        "review_status": "unreviewed",
        "best_frame_index": 3,
        "best_source_seconds": 0.6,
        "recorded_utc": "2026-10-11T14:06:02+00:00",
        "score": 0.8,
        "latitude": 53.8755,
        "longitude": -1.5386,
    }
    (tmp_path / "observations.json").write_text(json.dumps([observation]))

    target = generate_report(tmp_path, "/__roadlens_osm.pmtiles")

    page = target.read_text()
    assert '"evidence": null' in page
    assert "new pmtiles.PMTiles('/__roadlens_osm.pmtiles')" in page
    assert "Download reviewed GeoJSON" in page
