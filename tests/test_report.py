import hashlib
import json
import shutil
import subprocess

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


def _review_in_browser(page, storage=None, action=""):
    """Execute the page's storage logic without loading maps or network assets."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to exercise browser review storage")
    bootstrap = page.split("<script>\n", 1)[1].split("const markers =", 1)[0]
    set_status = (
        "function setStatus"
        + page.split("function setStatus", 1)[1].split("function render()", 1)[0]
    )
    script = (
        "const storage = " + json.dumps(storage or {}) + ";\n"
        "const localStorage = {getItem: key => storage[key] || null, "
        "setItem: (key, value) => storage[key] = value};\n"
        "const markers = new Map();\n"
        "const document = {getElementById: () => "
        "({querySelector: () => ({})})};\n"
        "function updateExport() {}\n"
        + bootstrap
        + set_status
        + action
        + "\nconsole.log(JSON.stringify({storage, "
        "statuses: observations.map(statusFor)}));"
    )
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_report_keeps_candidate_decisions_when_candidates_are_added(tmp_path):
    _write_run(tmp_path)
    page = generate_report(tmp_path, "/t").read_text()
    first = _review_in_browser(page, action="setStatus(observations[0], 'confirmed');")
    assert first["statuses"] == ["confirmed"]

    path = tmp_path / "observations.json"
    observations = json.loads(path.read_text())
    observations.append({**observations[0], "object_id": 2, "best_frame_index": 8})
    path.write_text(json.dumps(observations))
    expanded_page = generate_report(tmp_path, "/t").read_text()
    assert _storage_key(expanded_page) == _storage_key(page)
    expanded = _review_in_browser(expanded_page, first["storage"])
    assert expanded["statuses"] == ["confirmed", "unreviewed"]

    # Reusing an object ID for a different best frame must not inherit its status.
    observations[0]["best_frame_index"] = 9
    observations[0]["best_source_seconds"] = 3.0
    path.write_text(json.dumps(observations))
    replaced = _review_in_browser(
        generate_report(tmp_path, "/t").read_text(), expanded["storage"]
    )
    assert replaced["statuses"] == ["unreviewed", "unreviewed"]


def test_report_migrates_legacy_browser_decisions(tmp_path):
    _write_run(tmp_path)
    observations = json.loads((tmp_path / "observations.json").read_text())
    identity = [
        [o["object_id"], o["best_frame_index"], o["best_source_seconds"]]
        for o in observations
    ]
    old_id = hashlib.sha256(
        json.dumps([str(tmp_path.resolve()), identity]).encode("utf-8")
    ).hexdigest()[:16]
    storage = {f"roadlens-review-{old_id}": json.dumps({"1": "confirmed"})}
    page = generate_report(tmp_path, "/t").read_text()
    migrated = _review_in_browser(page, storage)
    assert migrated["statuses"] == ["confirmed"]
    changed = _review_in_browser(
        page, migrated["storage"], "setStatus(observations[0], 'rejected');"
    )
    assert _review_in_browser(page, changed["storage"])["statuses"] == ["rejected"]


def test_report_overlays_studio_decisions_without_rewriting_observations(tmp_path):
    _write_run(tmp_path)
    path = tmp_path / "observations.json"
    original = path.read_text()
    page = generate_report(tmp_path, "/t").read_text()
    browser = _review_in_browser(page, action="setStatus(observations[0], 'rejected');")
    (tmp_path / "review-decisions.json").write_text(json.dumps({"1": "confirmed"}))

    page = generate_report(tmp_path, "/t").read_text()
    embedded = json.loads(page.split("const observations = ", 1)[1].split(";\n", 1)[0])
    assert embedded[0]["review_status"] == "confirmed"
    imported = _review_in_browser(page, browser["storage"])
    assert imported["statuses"] == ["confirmed"]
    # Later browser reviews persist until Studio supplies a different decision.
    changed = _review_in_browser(
        page, imported["storage"], "setStatus(observations[0], 'rejected');"
    )
    assert _review_in_browser(page, changed["storage"])["statuses"] == ["rejected"]
    (tmp_path / "review-decisions.json").write_text(json.dumps({"1": "unreviewed"}))
    updated_page = generate_report(tmp_path, "/t").read_text()
    assert _review_in_browser(updated_page, changed["storage"])["statuses"] == [
        "unreviewed"
    ]
    assert path.read_text() == original
    assert "Points are camera GPS positions" in page
    assert "Garmin camera GPS" not in page
