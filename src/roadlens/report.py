# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoadLens review</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #101511; color: #edf2ec; }
    header { height: 64px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; border-bottom: 1px solid #344137; background: #172019; }
    header strong { font-size: 22px; letter-spacing: -.4px; }
    header span { color: #a8b5aa; font-size: 13px; }
    main { display: grid; grid-template-columns: 390px 1fr; height: calc(100vh - 64px); }
    aside { overflow-y: auto; padding: 20px; background: #172019; }
    #map { width: 100%; height: 100%; background: #263027; }
    .summary { color: #a8b5aa; font-size: 14px; line-height: 1.5; margin-bottom: 18px; }
    .card { background: #202b22; border: 1px solid #3b493e; border-radius: 12px; padding: 14px; margin-bottom: 14px; }
    .card.confirmed { border-color: #5ea86c; }
    .card.rejected { opacity: .58; }
    .card img { width: 100%; border-radius: 8px; margin: 10px 0; cursor: pointer; }
    .card h2 { font-size: 17px; margin: 0; text-transform: capitalize; }
    .meta { color: #a8b5aa; font-size: 12px; line-height: 1.6; }
    .actions { display: flex; gap: 8px; margin-top: 10px; }
    button { border: 1px solid #526157; border-radius: 7px; padding: 8px 11px; background: #2a372d; color: #edf2ec; cursor: pointer; }
    button:hover { background: #3b4b40; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    button.confirm { background: #286238; }
    button.reject { background: #63332e; }
    .empty { padding: 24px 12px; color: #a8b5aa; text-align: center; }
    .badge { float: right; color: #e8b44d; font-size: 11px; text-transform: uppercase; }
    @media (max-width: 800px) { main { grid-template-columns: 1fr; grid-template-rows: 45% 55%; } aside { order: 2; } }
  </style>
</head>
<body>
<header><strong>RoadLens</strong><span>Post-ride evidence review · camera GPS, not exact hazard positions</span></header>
<main>
  <aside>
    <div class="summary"><b id="count"></b> candidate observation(s), initially unreviewed.<br>Confirm or reject each candidate, then export only the human-confirmed GeoJSON.</div>
    <div class="actions" style="margin-bottom:18px"><button id="download" disabled>Download confirmed GeoJSON (0)</button><button id="downloadAudit">Full audit (all statuses)</button></div>
    <section id="observations"></section>
  </aside>
  <div id="map"></div>
</main>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/pmtiles@4.3.0/dist/pmtiles.js"></script>
<script>
const observations = __OBSERVATIONS__;
// Keyed per run so decisions never leak between runs served at the same origin.
const storageKey = 'roadlens-review-v2-__RUN_ID__';
const saved = JSON.parse(localStorage.getItem(storageKey) || '{}');
function candidateKey(item) {
  return JSON.stringify([item.object_id, item.best_frame_index, item.best_source_seconds]);
}
// The old key included the entire candidate list and stored statuses by object ID.
const legacyKey = 'roadlens-review-__LEGACY_RUN_ID__';
const legacy = JSON.parse(localStorage.getItem(legacyKey) || '{}');
const reviewDecisions = __REVIEW_DECISIONS__;
const importedDecisions = saved.studioDecisions || {};
observations.forEach(item => {
  const key = candidateKey(item);
  if (!saved[key] && legacy[item.object_id]) saved[key] = legacy[item.object_id];
  // Import changed Studio decisions while preserving subsequent browser reviews.
  const status = reviewDecisions[item.object_id];
  if (status && importedDecisions[key] !== status) saved[key] = status;
  importedDecisions[key] = status;
});
saved.studioDecisions = importedDecisions;
localStorage.setItem(storageKey, JSON.stringify(saved));
function statusFor(item) { return saved[candidateKey(item)] || item.review_status; }
const markers = new Map();
const map = L.map('map', {zoomControl: true});
const archive = new pmtiles.PMTiles('__TILE_URL__');
pmtiles.leafletRasterLayer(archive, {attribution: '© OpenStreetMap contributors', maxZoom: 19}).addTo(map);
const bounds = [];
function esc(val) {
  return String(val ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
function updateExport() {
  const confirmed = observations.filter(item => statusFor(item) === 'confirmed').length;
  const button = document.getElementById('download');
  button.textContent = 'Download confirmed GeoJSON (' + confirmed + ')';
  button.disabled = confirmed === 0;
}
function colour(status) { return status === 'confirmed' ? '#61ba73' : status === 'rejected' ? '#a25b52' : '#e8b44d'; }
function setStatus(item, status) {
  saved[candidateKey(item)] = status;
  localStorage.setItem(storageKey, JSON.stringify(saved));
  const card = document.getElementById('card-' + item.object_id);
  card.className = 'card ' + status;
  card.querySelector('.badge').textContent = status;
  if (markers.has(item.object_id)) markers.get(item.object_id).setStyle({color: colour(status), fillColor: colour(status)});
  updateExport();
}
function render() {
  const container = document.getElementById('observations');
  document.getElementById('count').textContent = observations.length;
  if (!observations.length) {
    container.innerHTML = '<div class="empty">No candidates found.<br>This is a valid negative model result.</div>';
    map.setView([54.5, -2.5], 6);
    return;
  }
  observations.forEach(item => {
    const status = statusFor(item);
    const card = document.createElement('article');
    card.id = 'card-' + item.object_id;
    card.className = 'card ' + status;
    const score = item.score === null ? 'not supplied' : Number(item.score).toFixed(3);
    card.innerHTML = '<span class="badge">' + esc(status) + '</span><h2>' + esc(item.prompt) + '</h2>' +
      '<div class="meta">SAM score: ' + esc(score) + '<br>' + esc(item.recorded_utc) + '<br>Source offset: ' + Number(item.best_source_seconds).toFixed(1) + ' s</div>' +
      (item.evidence ? '<a href="' + esc(item.evidence) + '" target="_blank"><img src="' + esc(item.evidence) + '" alt="SAM segmentation evidence"></a>' : '') +
      '<div class="actions"><button class="confirm">Confirm</button><button class="reject">Reject</button></div>';
    card.querySelector('.confirm').onclick = () => setStatus(item, 'confirmed');
    card.querySelector('.reject').onclick = () => setStatus(item, 'rejected');
    container.appendChild(card);
    if (item.latitude !== null && item.longitude !== null) {
      const point = [item.latitude, item.longitude];
      bounds.push(point);
      const marker = L.circleMarker(point, {radius: 10, weight: 3, color: colour(status), fillColor: colour(status), fillOpacity: .55}).addTo(map);
      marker.bindPopup('<b>' + esc(item.prompt) + '</b><br>Camera GPS · ' + esc(item.recorded_utc));
      marker.on('click', () => card.scrollIntoView({behavior: 'smooth', block: 'center'}));
      markers.set(item.object_id, marker);
    }
  });
  if (bounds.length) map.fitBounds(bounds, {padding: [80, 80], maxZoom: 17});
  else map.setView([54.5, -2.5], 6);
  updateExport();
}
function download(onlyConfirmed) {
  const reviewed = observations.map(item => ({...item, review_status: statusFor(item)}));
  const subset = onlyConfirmed ? reviewed.filter(item => item.review_status === 'confirmed') : reviewed;
  if (onlyConfirmed && !subset.length) return;
  const features = subset.map(item => ({
    type: 'Feature',
    geometry: item.latitude === null ? null : {type: 'Point', coordinates: [item.longitude, item.latitude]},
    properties: item
  }));
  const payload = {
    type: 'FeatureCollection',
    metadata: {
      export_mode: onlyConfirmed ? 'human_confirmed_only' : 'full_audit',
      location_note: 'Points are camera GPS positions, not surveyed defect coordinates.'
    },
    features
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/geo+json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = onlyConfirmed ? 'roadlens-confirmed-hazards.geojson' : 'roadlens-audit-all.geojson';
  link.click();
  URL.revokeObjectURL(link.href);
}
document.getElementById('download').onclick = () => download(true);
document.getElementById('downloadAudit').onclick = () => download(false);
render();
</script>
</body>
</html>
"""


def generate_report(output_directory: Path, tile_url: str) -> Path:
    observations = json.loads(
        (output_directory / "observations.json").read_text(encoding="utf-8")
    )
    decisions_path = output_directory / "review-decisions.json"
    decisions = (
        json.loads(decisions_path.read_text(encoding="utf-8"))
        if decisions_path.is_file()
        else {}
    )
    for index, observation in enumerate(observations):
        key = str(int(observation.get("object_id", index)))
        if key in decisions:
            observation["review_status"] = decisions[key]
        evidence = f"overlay-{observation['best_frame_index']:06d}.png"
        observation["evidence"] = (
            evidence if (output_directory / evidence).is_file() else None
        )
    payload = json.dumps(observations).replace("</", "<\\/")
    # Keep the run key stable when candidates are added; identify each separately.
    run_path = str(output_directory.resolve())
    run_id = hashlib.sha256(run_path.encode("utf-8")).hexdigest()[:16]
    identity = [
        [o.get("object_id"), o.get("best_frame_index"), o.get("best_source_seconds")]
        for o in observations
    ]
    legacy_run_id = hashlib.sha256(
        json.dumps([run_path, identity]).encode("utf-8")
    ).hexdigest()[:16]
    # Substitute observation data last so its text cannot inject placeholders.
    page = (
        _PAGE.replace("__TILE_URL__", tile_url)
        .replace("__RUN_ID__", run_id)
        .replace("__LEGACY_RUN_ID__", legacy_run_id)
        .replace("__REVIEW_DECISIONS__", json.dumps(decisions).replace("</", "<\\/"))
        .replace("__OBSERVATIONS__", payload)
    )
    target = output_directory / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


def serve_report(
    output_directory: Path, pmtiles_path: Path, host: str, port: int
) -> None:
    if not pmtiles_path.is_file():
        raise FileNotFoundError(pmtiles_path)
    generate_report(output_directory, "/__roadlens_osm.pmtiles")
    handler = partial(
        _ReportHandler, directory=str(output_directory), pmtiles_path=pmtiles_path
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"RoadLens review: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class _ReportHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, pmtiles_path: Path, **kwargs):
        self.pmtiles_path = pmtiles_path
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/__roadlens_osm.pmtiles":
            self._serve_archive(send_body=True)
        else:
            super().do_GET()

    def do_HEAD(self):
        if self.path == "/__roadlens_osm.pmtiles":
            self._serve_archive(send_body=False)
        else:
            super().do_HEAD()

    def _serve_archive(self, send_body: bool) -> None:
        size = self.pmtiles_path.stat().st_size
        try:
            byte_range = parse_byte_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_error(416, "Invalid byte range")
            return
        start, end = byte_range or (0, size - 1)
        self.send_response(206 if byte_range else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if byte_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return
        with self.pmtiles_path.open("rb") as archive:
            archive.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = archive.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or (not match.group(1) and not match.group(2)):
        raise ValueError("Invalid byte range")
    if not match.group(1):
        length = int(match.group(2))
        if length <= 0:
            raise ValueError("Invalid suffix length")
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Byte range outside archive")
    return start, min(end, size - 1)
