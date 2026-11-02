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
    assert "Download confirmed GeoJSON" in page
    assert "'human_confirmed_only'" in page
    assert "filter(item => item.review_status === 'confirmed')" in page


def _write_run(directory, prompt="pothole", object_id=1):
    directory.mkdir(parents=True, exist_ok=True)
    observation = {
        "object_id": object_id,
        "prompt": prompt,
        "review_status": "unreviewed",
        "best_frame_index": 0,
        "best_source_seconds": 1.0,
        "recorded_utc": "2026-10-11T14:06:02+00:00",
        "score": 0.5,
        "latitude": None,
        "longitude": None,
    }
    (directory / "observations.json").write_text(json.dumps([observation]))


def _storage_key(page):
    marker = "const storageKey = 'roadlens-review-"
    return page.split(marker, 1)[1].split("'", 1)[0]


def test_report_review_storage_is_scoped_per_run(tmp_path):
    _write_run(tmp_path / "a")
    _write_run(tmp_path / "b")
    key_a = _storage_key(generate_report(tmp_path / "a", "/t").read_text())
    key_b = _storage_key(generate_report(tmp_path / "b", "/t").read_text())
    assert key_a != key_b
    assert "location.pathname" not in (tmp_path / "a" / "index.html").read_text()

    # VLM triage rewriting observation fields must not reset saved decisions.
    observations = json.loads((tmp_path / "a" / "observations.json").read_text())
    observations[0]["vlm_triage"] = {"classification": "uncertain"}
    (tmp_path / "a" / "observations.json").write_text(json.dumps(observations))
    assert _storage_key(generate_report(tmp_path / "a", "/t").read_text()) == key_a


def test_report_does_not_inject_unescaped_prompt_markup(tmp_path):
    _write_run(tmp_path, prompt='<img src=x onerror="alert(1)"> __RUN_ID__')
    page = generate_report(tmp_path, "/t").read_text()
    assert "esc(item.prompt)" in page
    assert "+ item.prompt +" not in page
    assert "__RUN_ID__" in page.split("const observations = ", 1)[1]
