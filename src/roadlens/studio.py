# ruff: noqa: E501
"""Interactive map-first post-ride workflow with bookended human-in-the-loop review."""

from __future__ import annotations

import gc
import json
import os
import threading
import time
import uuid
from argparse import Namespace
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from roadlens.pmtiles_roads import PmtilesStreetIndex
from roadlens.report import parse_byte_range
from roadlens.ride import (
    HAZARD_PRESETS,
    RideIndex,
    index_ride,
    resolve_by_corridor,
    resolve_by_ride_offset,
    resolve_by_street_name,
    resolve_by_wall_clock,
)
from roadlens.vlm import (
    create_vlm_backend,
    format_deterministic_summary_report,
    triage_run_directory,
)

STUDIO_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoadLens Studio · Post-Ride Workflow</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #0f1411; color: #edf2ec; }
    header { height: 60px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; border-bottom: 1px solid #2d3a30; background: #151d17; }
    header .brand { display: flex; align-items: baseline; gap: 12px; }
    header strong { font-size: 20px; letter-spacing: -0.4px; }
    header .sub { color: #9cb0a0; font-size: 12px; }
    main { display: grid; grid-template-columns: 440px 1fr; height: calc(100vh - 60px); }
    aside { overflow-y: auto; padding: 16px; background: #151d17; border-right: 1px solid #2d3a30; }
    #map { width: 100%; height: 100%; background: #1f2922; }
    .panel { background: #1c261f; border: 1px solid #314035; border-radius: 10px; padding: 13px; margin-bottom: 13px; }
    .panel h2 { margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.4px; color: #b7c9ba; }
    .muted { color: #9cb0a0; font-size: 12px; line-height: 1.5; }
    label { display: block; font-size: 12px; color: #b7c9ba; margin: 8px 0 4px; }
    input[type="text"], input[type="number"], select, textarea {
      width: 100%; background: #111813; border: 1px solid #3a4a3e; color: #edf2ec;
      border-radius: 7px; padding: 7px 9px; font-size: 13px; font-family: inherit;
    }
    textarea { min-height: 150px; resize: vertical; line-height: 1.45; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }
    button {
      border: 1px solid #495b4e; border-radius: 7px; padding: 7px 11px;
      background: #263429; color: #edf2ec; cursor: pointer; font-size: 12px; font-weight: 500;
    }
    button:hover:not(:disabled) { background: #344739; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    button.primary { background: #2b6b3e; border-color: #46915c; }
    button.primary:hover:not(:disabled) { background: #35804b; }
    button.confirm { background: #286238; border-color: #4b8f5d; }
    button.reject { background: #63332e; border-color: #96524b; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .chip { padding: 5px 9px; border-radius: 999px; font-size: 11px; background: #222e25; border: 1px solid #3c4d40; cursor: pointer; }
    .chip.active { background: #2b6b3e; border-color: #62b87b; color: #fff; }
    .dropzone { border: 1px dashed #495b4e; border-radius: 8px; padding: 10px; text-align: center; font-size: 12px; color: #9cb0a0; margin-top: 8px; cursor: pointer; }
    .dropzone.drag { border-color: #62b87b; background: #1e2f23; }
    .pass-item { background: #131b15; border: 1px solid #2f3e33; border-radius: 8px; padding: 8px 10px; margin-top: 7px; font-size: 12px; }
    .card { background: #131b15; border: 1px solid #344438; border-radius: 10px; padding: 11px; margin-top: 10px; }
    .card.confirmed { border-color: #5ea86c; }
    .card.rejected { opacity: 0.56; border-color: #7d4741; }
    .card img { width: 100%; border-radius: 7px; margin: 8px 0; cursor: pointer; }
    .badge { float: right; font-size: 10px; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; background: #2a362d; color: #e8b44d; }
    .triage-pill { display: inline-block; font-size: 11px; padding: 2px 7px; border-radius: 5px; margin-top: 4px; background: #243128; color: #cde2d1; }
    .warn { background: #332616; border: 1px solid #6e5227; color: #f2cf8d; padding: 8px 10px; border-radius: 7px; font-size: 11px; margin-top: 8px; line-height: 1.4; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; grid-template-rows: 48% 52%; } aside { order: 2; } }
  </style>
</head>
<body>
<header>
  <div class="brand">
    <strong>RoadLens Studio</strong>
    <span class="sub">Bookended Human-in-the-Loop · Scope on OSM -> Targeted Scan/VLM -> Confirm & Export</span>
  </div>
  <span class="sub" id="rideStatus">No ride indexed yet</span>
</header>
<main>
  <aside>
    <div class="panel">
      <h2>1. Load Ride Footage or Existing Scan</h2>
      <label>Local Garmin clip (.MP4) or ride directory path</label>
      <div class="row">
        <input id="ridePath" type="text" placeholder="/path/to/GRMS0010.MP4 or folder" value="__INITIAL_RIDE_PATH__">
        <button id="loadRideBtn" class="primary" style="flex:0 0 auto">Index GPS</button>
      </div>
      <div id="dropzone" class="dropzone">
        Or drop / click to import a local .MP4 clip into workspace<input id="fileInput" type="file" accept=".mp4,.MP4" style="display:none">
      </div>
      <label style="margin-top:10px">Or open an existing scan/analyze output folder</label>
      <div class="row">
        <input id="existingRunPath" type="text" placeholder="artifacts/eccup-survey/return">
        <button id="openRunBtn" style="flex:0 0 auto">Open Run</button>
      </div>
    </div>

    <div class="panel">
      <h2>2. Scope Road Fragment & Hazard</h2>
      <div class="muted">Click two points along the blue GPS track on the map to select a road fragment (finds outbound + return passes), or resolve by street / time below:</div>
      <div class="row" style="margin-top:8px">
        <select id="queryMode" style="flex:0 0 135px">
          <option value="street">Street name</option>
          <option value="offset">Ride offset</option>
          <option value="clock">Wall clock</option>
        </select>
        <input id="queryValue" type="text" placeholder="e.g. Eccup Moor Road, 20m, or 12:30">
        <button id="resolveQueryBtn" style="flex:0 0 auto">Find</button>
      </div>
      <div class="row" style="margin-top:6px">
        <button id="clearMapSelBtn">Clear map selection</button>
      </div>
      <div id="resolvedPasses"></div>

      <label style="margin-top:10px">Hazard preset or custom description</label>
      <div id="presetChips" class="chips"></div>
      <input id="hazardPrompt" type="text" value="pothole" style="margin-top:6px" placeholder="Describe hazard (e.g. fallen tree branch)">

      <div class="row" style="margin-top:8px">
        <div>
          <label>SAM 3.1 mode</label>
          <select id="samMode">
            <option value="scan_local">Local distance scan (~1m + refine)</option>
            <option value="scan_coarse">Local coarse scan (~1m only)</option>
            <option value="analyze_remote">Remote API clip pass</option>
          </select>
        </div>
        <div>
          <label>VLM triage & report</label>
          <select id="vlmMode">
            <option value="local">Local Muse Glimmer</option>
            <option value="none">Skip VLM (fast template)</option>
            <option value="remote">Remote API VLM</option>
          </select>
        </div>
      </div>
      <div id="remotePrivacyWarn" class="warn" style="display:none">
        Remote API notice: selected video frames (with bottom Garmin GPS bar masked on VLM calls) will be sent to the configured remote API endpoint.
      </div>
      <div class="row" style="margin-top:10px">
        <button id="runScanBtn" class="primary" disabled>Scan Selected Pass(es)</button>
      </div>
      <div id="jobStatus" class="muted" style="margin-top:8px"></div>
    </div>

    <div class="panel">
      <h2>3. Human Review & Summary Report</h2>
      <div class="muted" id="reviewSummary">Load or run a targeted scan to review candidate overlays. Unreviewed model candidates are never exported as confirmed hazards.</div>
      <div class="row" style="margin-top:8px">
        <button id="triageRunBtn" disabled>Run VLM Triage</button>
        <button id="draftReportBtn" disabled>Draft Summary Report</button>
      </div>
      <div id="reportSection" style="display:none; margin-top:10px">
        <label>Editable Summary Report (verify before manual council / FixMyStreet submission)</label>
        <textarea id="reportEditor"></textarea>
        <div class="row" style="margin-top:6px">
          <button id="saveReportBtn">Save Draft</button>
          <button id="downloadReportBtn">Download .md</button>
        </div>
      </div>
      <div class="row" style="margin-top:10px">
        <button id="exportConfirmedBtn" class="primary" disabled>Export Confirmed GeoJSON (0)</button>
        <button id="exportAuditBtn" disabled>Full Audit GeoJSON</button>
      </div>
      <div id="candidateCards"></div>
    </div>
  </aside>
  <div id="map"></div>
</main>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/pmtiles@4.3.0/dist/pmtiles.js"></script>
<script src="https://unpkg.com/protomaps-leaflet@5.0.0/dist/protomaps-leaflet.js"></script>
<script>
const PRESETS = __HAZARD_PRESETS__;
const map = L.map('map', {zoomControl: true}).setView([53.85, -1.54], 12);
try {
  if (window.protomapsL && protomapsL.leafletLayer) {
    protomapsL.leafletLayer({url: '/__roadlens_osm.pmtiles', flavor: 'dark', attribution: '© OpenStreetMap contributors'}).addTo(map);
  } else {
    const archive = new pmtiles.PMTiles('/__roadlens_osm.pmtiles');
    pmtiles.leafletRasterLayer(archive, {attribution: '© OpenStreetMap contributors', maxZoom: 19}).addTo(map);
  }
} catch (err) {
  console.warn('Basemap layer fallback:', err);
}

const rideLayer = L.layerGroup().addTo(map);
const passLayer = L.layerGroup().addTo(map);
const clickLayer = L.layerGroup().addTo(map);
const candidateLayer = L.layerGroup().addTo(map);
const candidateMarkers = new Map();

let currentPasses = [];
let mapClicks = [];
let activeRun = null;

function esc(val) {
  return String(val ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderPresets() {
  const wrap = document.getElementById('presetChips');
  const input = document.getElementById('hazardPrompt');
  PRESETS.forEach((p, idx) => {
    const el = document.createElement('span');
    el.className = 'chip' + (idx === 0 ? ' active' : '');
    el.textContent = p.label;
    el.title = p.description;
    el.onclick = () => {
      document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
      el.classList.add('active');
      input.value = p.sam_prompt;
    };
    wrap.appendChild(el);
  });
}
renderPresets();

function updatePrivacyBanner() {
  const sam = document.getElementById('samMode').value;
  const vlm = document.getElementById('vlmMode').value;
  const isRemote = sam.includes('remote') || vlm === 'remote';
  document.getElementById('remotePrivacyWarn').style.display = isRemote ? 'block' : 'none';
}
document.getElementById('samMode').onchange = updatePrivacyBanner;
document.getElementById('vlmMode').onchange = updatePrivacyBanner;

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: {'Content-Type': 'application/json', ...(opts.headers || {})},
    ...opts
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}

function drawRideTrack(ridePayload) {
  rideLayer.clearLayers();
  passLayer.clearLayers();
  mapClicks = [];
  clickLayer.clearLayers();
  const summary = ridePayload.ride;
  document.getElementById('rideStatus').textContent =
    summary.primary_clip_count + ' clip(s) · ' +
    (summary.total_distance_m / 1000).toFixed(1) + ' km · ' +
    Math.round(summary.total_elapsed_seconds / 60) + ' min';
  const geo = L.geoJSON(ridePayload.geojson, {
    style: {color: '#4da3ff', weight: 4, opacity: 0.82}
  }).addTo(rideLayer);
  geo.on('click', (ev) => {
    L.DomEvent.stopPropagation(ev);
    handleMapPoint(ev.latlng);
  });
  if (ridePayload.geojson.features.length) {
    map.fitBounds(geo.getBounds(), {padding: [40, 40]});
  }
}

async function handleMapPoint(latlng) {
  if (mapClicks.length >= 2) {
    mapClicks = [];
    clickLayer.clearLayers();
  }
  mapClicks.push([latlng.lat, latlng.lng]);
  L.circleMarker(latlng, {radius: 6, color: '#ff9f43', fillColor: '#ff9f43', fillOpacity: 0.9}).addTo(clickLayer);
  if (mapClicks.length === 2) {
    L.polyline(mapClicks, {color: '#ff9f43', dashArray: '5,5', weight: 2}).addTo(clickLayer);
    const data = await api('/api/resolve', {
      method: 'POST',
      body: JSON.stringify({mode: 'corridor', polyline: mapClicks})
    });
    setResolvedPasses(data.passes);
  }
}

map.on('click', (ev) => {
  if (rideLayer.getLayers().length) handleMapPoint(ev.latlng);
});

document.getElementById('clearMapSelBtn').onclick = () => {
  mapClicks = [];
  currentPasses = [];
  clickLayer.clearLayers();
  passLayer.clearLayers();
  document.getElementById('resolvedPasses').innerHTML = '';
  document.getElementById('runScanBtn').disabled = true;
};

function setResolvedPasses(passes) {
  currentPasses = passes;
  passLayer.clearLayers();
  const wrap = document.getElementById('resolvedPasses');
  if (!passes.length) {
    wrap.innerHTML = '<div class="warn">No GPS track segments matched that selection.</div>';
    document.getElementById('runScanBtn').disabled = true;
    return;
  }
  const allBounds = [];
  wrap.innerHTML = passes.map((p, idx) => {
    if (p.polyline && p.polyline.length > 1) {
      const line = L.polyline(p.polyline, {color: '#ff6b6b', weight: 6, opacity: 0.9}).addTo(passLayer);
      allBounds.push(...p.polyline);
      line.bindPopup('Pass ' + esc(p.pass_index) + ' · ' + esc(p.clip_name));
    }
    const street = [p.street_name, p.place_name].filter(Boolean).join(', ') || 'Unnamed road';
    return '<div class="pass-item">' +
      '<label style="margin:0"><input type="checkbox" class="passCheck" data-idx="' + idx + '" checked> ' +
      '<b>Pass ' + esc(p.pass_index) + ': ' + esc(p.clip_name) + '</b> (' + p.start_seconds.toFixed(1) + '–' + p.end_seconds.toFixed(1) + ' s, ' +
      Math.round(p.distance_m) + ' m ' + esc(p.compass_direction || '') + ')</label>' +
      '<div class="muted">' + esc(street) + ' · ~' + p.estimated_coarse_frames_1m + ' coarse / ' + p.estimated_native_frames + ' native frames</div>' +
      '</div>';
  }).join('');
  if (allBounds.length) map.fitBounds(allBounds, {padding: [60, 60], maxZoom: 16});
  document.getElementById('runScanBtn').disabled = false;
}

document.getElementById('loadRideBtn').onclick = async () => {
  const path = document.getElementById('ridePath').value.trim();
  if (!path) return;
  try {
    document.getElementById('rideStatus').textContent = 'Indexing Garmin telemetry...';
    const data = await api('/api/index-ride', {method: 'POST', body: JSON.stringify({path})});
    drawRideTrack(data);
  } catch (err) {
    alert('Could not index ride: ' + err.message);
  }
};

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
dropzone.onclick = () => fileInput.click();
dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('drag'); };
dropzone.ondragleave = () => dropzone.classList.remove('drag');
dropzone.ondrop = async (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  if (e.dataTransfer.files.length) await uploadClip(e.dataTransfer.files[0]);
};
fileInput.onchange = async () => {
  if (fileInput.files.length) await uploadClip(fileInput.files[0]);
};

async function uploadClip(file) {
  document.getElementById('rideStatus').textContent = 'Importing ' + file.name + '...';
  const res = await fetch('/api/upload-clip', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/octet-stream',
      'X-Filename': encodeURIComponent(file.name)
    },
    body: file
  });
  const data = await res.json();
  if (!res.ok) { alert(data.error || 'Upload failed'); return; }
  document.getElementById('ridePath').value = data.saved_path;
  drawRideTrack(data);
}

document.getElementById('resolveQueryBtn').onclick = async () => {
  const mode = document.getElementById('queryMode').value;
  const value = document.getElementById('queryValue').value.trim();
  if (!value) return;
  try {
    const data = await api('/api/resolve', {method: 'POST', body: JSON.stringify({mode, value})});
    setResolvedPasses(data.passes);
  } catch (err) {
    alert('Resolve error: ' + err.message);
  }
};

document.getElementById('openRunBtn').onclick = async () => {
  const path = document.getElementById('existingRunPath').value.trim();
  if (!path) return;
  try {
    const run = await api('/api/open-run', {method: 'POST', body: JSON.stringify({path})});
    renderRun(run);
  } catch (err) {
    alert('Open run failed: ' + err.message);
  }
};

document.getElementById('runScanBtn').onclick = async () => {
  const checks = [...document.querySelectorAll('.passCheck:checked')];
  const chosen = checks.map(c => currentPasses[Number(c.dataset.idx)]);
  if (!chosen.length) { alert('Select at least one pass.'); return; }
  const prompt = document.getElementById('hazardPrompt').value.trim() || 'pothole';
  const sam_mode = document.getElementById('samMode').value;
  const vlm_mode = document.getElementById('vlmMode').value;
  document.getElementById('runScanBtn').disabled = true;
  document.getElementById('jobStatus').textContent = 'Starting targeted job...';
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({passes: chosen, prompt, sam_mode, vlm_mode})
    });
    pollJob(job.job_id);
  } catch (err) {
    document.getElementById('jobStatus').textContent = 'Error: ' + err.message;
    document.getElementById('runScanBtn').disabled = false;
  }
};

async function pollJob(jobId) {
  const statusEl = document.getElementById('jobStatus');
  const tick = async () => {
    const job = await api('/api/jobs/' + jobId);
    statusEl.textContent = '[' + job.status + '] ' + (job.progress || '');
    if (job.status === 'complete' && job.run) {
      document.getElementById('runScanBtn').disabled = false;
      renderRun(job.run);
      return;
    }
    if (job.status === 'failed') {
      document.getElementById('runScanBtn').disabled = false;
      return;
    }
    setTimeout(tick, 1500);
  };
  await tick();
}

function markerColour(status) {
  return status === 'confirmed' ? '#61ba73' : status === 'rejected' ? '#a25b52' : '#e8b44d';
}

function updateCounts() {
  if (!activeRun) return;
  const obs = activeRun.observations;
  const confirmed = obs.filter(o => o.review_status === 'confirmed').length;
  const rejected = obs.filter(o => o.review_status === 'rejected').length;
  const unreviewed = obs.length - confirmed - rejected;
  document.getElementById('reviewSummary').innerHTML =
    '<b>' + obs.length + '</b> candidate(s) in <code>' + esc(activeRun.run_name) + '</code>: ' +
    '<b>' + confirmed + '</b> confirmed · <b>' + unreviewed + '</b> unreviewed · <b>' + rejected + '</b> rejected.';
  const btn = document.getElementById('exportConfirmedBtn');
  btn.textContent = 'Export Confirmed GeoJSON (' + confirmed + ')';
  btn.disabled = confirmed === 0;
  document.getElementById('exportAuditBtn').disabled = obs.length === 0;
  document.getElementById('triageRunBtn').disabled = obs.length === 0;
  document.getElementById('draftReportBtn').disabled = false;
}

function renderRun(run) {
  activeRun = run;
  candidateLayer.clearLayers();
  candidateMarkers.clear();
  const container = document.getElementById('candidateCards');
  container.innerHTML = '';
  const bounds = [];

  if (run.summary_report) {
    document.getElementById('reportSection').style.display = 'block';
    document.getElementById('reportEditor').value = run.summary_report;
  } else {
    document.getElementById('reportSection').style.display = 'none';
  }

  run.observations.forEach(item => {
    const card = document.createElement('article');
    card.id = 'cand-' + item.object_id;
    card.className = 'card ' + item.review_status;
    const score = item.score === null || item.score === undefined ? 'n/a' : Number(item.score).toFixed(3);
    const triage = item.vlm_triage;
    const triageHtml = triage
      ? '<div class="triage-pill">VLM: <b>' + esc(triage.classification) + '</b> (' + esc(triage.confidence) + ')</div>' +
        '<div class="muted" style="margin-top:4px">' + esc(triage.visual_description || '') + '</div>'
      : '';
    const imgUrl = item.overlay_url || item.frame_url;
    card.innerHTML =
      '<span class="badge">' + esc(item.review_status) + '</span>' +
      '<strong>#' + esc(item.object_id) + ' · ' + esc(item.prompt) + '</strong>' +
      '<div class="muted">SAM score: ' + score + ' · t=' + Number(item.best_source_seconds).toFixed(2) + 's</div>' +
      triageHtml +
      (imgUrl ? '<a href="' + esc(imgUrl) + '" target="_blank"><img src="' + esc(imgUrl) + '" loading="lazy" alt="Candidate evidence"></a>' : '') +
      '<div class="row"><button class="confirm">Confirm Hazard</button><button class="reject">Reject (Patch/Debris)</button></div>';

    card.querySelector('.confirm').onclick = () => setDecision(item, 'confirmed');
    card.querySelector('.reject').onclick = () => setDecision(item, 'rejected');
    container.appendChild(card);

    if (item.latitude !== null && item.longitude !== null) {
      const pt = [item.latitude, item.longitude];
      bounds.push(pt);
      const m = L.circleMarker(pt, {
        radius: 9, weight: 2,
        color: markerColour(item.review_status),
        fillColor: markerColour(item.review_status),
        fillOpacity: 0.65
      }).addTo(candidateLayer);
      m.bindPopup('<b>#' + esc(item.object_id) + ' ' + esc(item.prompt) + '</b><br>Camera GPS · ' + Number(item.best_source_seconds).toFixed(1) + 's');
      m.on('click', () => card.scrollIntoView({behavior: 'smooth', block: 'center'}));
      candidateMarkers.set(item.object_id, m);
    }
  });
  if (bounds.length) map.fitBounds(bounds, {padding: [60, 60], maxZoom: 17});
  updateCounts();
}

async function setDecision(item, status) {
  const res = await api('/api/runs/' + activeRun.run_id + '/review', {
    method: 'POST',
    body: JSON.stringify({object_id: item.object_id, review_status: status})
  });
  item.review_status = res.review_status;
  const card = document.getElementById('cand-' + item.object_id);
  card.className = 'card ' + status;
  card.querySelector('.badge').textContent = status;
  if (candidateMarkers.has(item.object_id)) {
    const col = markerColour(status);
    candidateMarkers.get(item.object_id).setStyle({color: col, fillColor: col});
  }
  updateCounts();
}

document.getElementById('triageRunBtn').onclick = async () => {
  if (!activeRun) return;
  const vlm = document.getElementById('vlmMode').value === 'none' ? 'local' : document.getElementById('vlmMode').value;
  document.getElementById('jobStatus').textContent = 'Running VLM triage...';
  try {
    const run = await api('/api/runs/' + activeRun.run_id + '/triage', {
      method: 'POST', body: JSON.stringify({vlm_mode: vlm})
    });
    document.getElementById('jobStatus').textContent = 'VLM triage complete.';
    renderRun(run);
  } catch (err) {
    alert('VLM triage failed: ' + err.message);
  }
};

document.getElementById('draftReportBtn').onclick = async () => {
  if (!activeRun) return;
  const vlm = document.getElementById('vlmMode').value;
  const res = await api('/api/runs/' + activeRun.run_id + '/draft-report', {
    method: 'POST', body: JSON.stringify({vlm_mode: vlm})
  });
  document.getElementById('reportSection').style.display = 'block';
  document.getElementById('reportEditor').value = res.summary_report;
};

document.getElementById('saveReportBtn').onclick = async () => {
  if (!activeRun) return;
  const content = document.getElementById('reportEditor').value;
  await api('/api/runs/' + activeRun.run_id + '/save-report', {
    method: 'POST', body: JSON.stringify({summary_report: content})
  });
};

document.getElementById('downloadReportBtn').onclick = () => {
  const text = document.getElementById('reportEditor').value;
  const blob = new Blob([text], {type: 'text/markdown'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (activeRun ? activeRun.run_name : 'roadlens') + '-summary-report.md';
  a.click();
  URL.revokeObjectURL(a.href);
};

function downloadGeojson(onlyConfirmed) {
  if (!activeRun) return;
  const subset = onlyConfirmed
    ? activeRun.observations.filter(o => o.review_status === 'confirmed')
    : activeRun.observations;
  if (onlyConfirmed && !subset.length) {
    alert('Confirm at least one candidate before exporting council-ready GeoJSON.');
    return;
  }
  const features = subset.map(o => ({
    type: 'Feature',
    geometry: o.latitude === null ? null : {type: 'Point', coordinates: [o.longitude, o.latitude]},
    properties: o
  }));
  const payload = {
    type: 'FeatureCollection',
    metadata: {
      export_mode: onlyConfirmed ? 'human_confirmed_only' : 'full_audit',
      location_note: 'Points are Garmin camera GPS positions, not surveyed defect coordinates.'
    },
    features
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/geo+json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = onlyConfirmed ? 'roadlens-confirmed-hazards.geojson' : 'roadlens-audit-all.geojson';
  a.click();
  URL.revokeObjectURL(a.href);
}
document.getElementById('exportConfirmedBtn').onclick = () => downloadGeojson(true);
document.getElementById('exportAuditBtn').onclick = () => downloadGeojson(false);

if (document.getElementById('ridePath').value.trim()) {
  document.getElementById('loadRideBtn').click();
}
</script>
</body>
</html>
"""


class StudioState:
    def __init__(
        self,
        workspace_dir: Path,
        pmtiles_path: Path,
        checkpoint_path: Path | None = None,
        initial_ride_path: Path | None = None,
    ):
        if not pmtiles_path.is_file():
            raise FileNotFoundError(pmtiles_path)
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.pmtiles_path = pmtiles_path
        self.checkpoint_path = checkpoint_path
        self.initial_ride_path = initial_ride_path
        self.street_index = PmtilesStreetIndex(pmtiles_path)
        self.ride: RideIndex | None = None
        self.runs: dict[str, Path] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register_run(self, run_dir: Path) -> str:
        resolved = run_dir.resolve()
        if not (resolved / "observations.json").is_file():
            raise FileNotFoundError(f"No observations.json in {run_dir}")
        with self._lock:
            for existing_id, existing_path in self.runs.items():
                if existing_path == resolved:
                    return existing_id
            run_id = uuid.uuid4().hex[:10]
            self.runs[run_id] = resolved
            return run_id

    def serialize_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.runs[run_id]
        obs_path = run_dir / "observations.json"
        observations: list[dict[str, Any]] = json.loads(
            obs_path.read_text(encoding="utf-8")
        )
        decisions_path = run_dir / "review-decisions.json"
        decisions: dict[str, str] = {}
        if decisions_path.is_file():
            decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
        enriched = []
        for idx, item in enumerate(observations):
            copy_item = dict(item)
            oid = int(copy_item.get("object_id", idx))
            if str(oid) in decisions:
                copy_item["review_status"] = decisions[str(oid)]
            fidx = int(copy_item.get("best_frame_index", oid))
            ov_name = f"overlay-{fidx:06d}.png"
            fr_name = f"frame-{fidx:06d}.png"
            copy_item["overlay_url"] = (
                f"/runs/{run_id}/files/{ov_name}"
                if (run_dir / ov_name).is_file()
                else None
            )
            copy_item["frame_url"] = (
                f"/runs/{run_id}/files/{fr_name}"
                if (run_dir / fr_name).is_file()
                else None
            )
            enriched.append(copy_item)
        report_path = run_dir / "summary-report.md"
        summary_report = (
            report_path.read_text(encoding="utf-8") if report_path.is_file() else None
        )
        return {
            "run_id": run_id,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "observations": enriched,
            "summary_report": summary_report,
        }

    def update_review_decision(
        self, run_id: str, object_id: int, review_status: str
    ) -> str:
        if review_status not in ("confirmed", "rejected", "unreviewed"):
            raise ValueError(f"Invalid review_status: {review_status!r}")
        run_dir = self.runs[run_id]
        decisions_path = run_dir / "review-decisions.json"
        with self._lock:
            decisions: dict[str, str] = (
                json.loads(decisions_path.read_text(encoding="utf-8"))
                if decisions_path.is_file()
                else {}
            )
            decisions[str(int(object_id))] = review_status
            decisions_path.write_text(
                json.dumps(decisions, indent=2) + "\n", encoding="utf-8"
            )
        return review_status


def render_studio_html(initial_ride_path: Path | None = None) -> str:
    import html

    presets_json = json.dumps(list(HAZARD_PRESETS)).replace("</", "<\\/")
    init_str = html.escape(str(initial_ride_path), quote=True) if initial_ride_path else ""
    return STUDIO_HTML.replace("__HAZARD_PRESETS__", presets_json).replace(
        "__INITIAL_RIDE_PATH__", init_str
    )


def _execute_targeted_job(
    state: StudioState,
    job_id: str,
    passes: list[dict[str, Any]],
    prompt: str,
    sam_mode: str,
    vlm_mode: str,
) -> None:
    try:
        from roadlens.analysis import (
            Observation,
            build_observations,
            write_observations,
        )
        from roadlens.backends import create_backend
        from roadlens.media import prepare_analysis_clip, probe_video
        from roadlens.output import write_frame_result
        from roadlens.scan import run_scan
        from roadlens.telemetry import read_garmin_telemetry

        stamp = time.strftime("%Y%m%d-%H%M%S")
        job_dir = state.workspace_dir / f"job-{stamp}-{job_id[:6]}"
        job_dir.mkdir(parents=True, exist_ok=True)

        merged_obs: list[Observation] = []
        street_names: list[str] = []
        place_names: list[str] = []

        for p_idx, seg in enumerate(passes):
            clip_path = Path(seg["clip_path"])
            start_s = float(seg["start_seconds"])
            end_s = float(seg["end_seconds"])
            dur_s = max(1.0, end_s - start_s)
            if seg.get("street_name"):
                street_names.append(str(seg["street_name"]))
            if seg.get("place_name"):
                place_names.append(str(seg["place_name"]))

            pass_dir = job_dir if len(passes) == 1 else (job_dir / f"pass-{p_idx + 1:02d}")
            state.jobs[job_id]["progress"] = (
                f"Pass {p_idx + 1}/{len(passes)}: {clip_path.name} "
                f"({start_s:.1f}–{end_s:.1f}s) for '{prompt}'"
            )

            if sam_mode == "analyze_remote":
                from datetime import datetime

                meta = probe_video(clip_path)
                created = datetime.fromisoformat(
                    meta["creation_time"].replace("Z", "+00:00")
                )
                tel = read_garmin_telemetry(clip_path)
                pass_dir.mkdir(parents=True, exist_ok=True)
                working = pass_dir / "analysis.mp4"
                prepare_analysis_clip(clip_path, working, start_s, dur_s, 5.0)
                backend = create_backend("remote", confidence_threshold=0.35)
                frames = []
                for frame_res in backend.segment_video(working, prompt):
                    frames.append(frame_res)
                    if frame_res.detections:
                        write_frame_result(working, frame_res, pass_dir)
                obs_list = build_observations(frames, tel.samples, created, start_s, 5.0)
                write_observations(obs_list, clip_path, pass_dir)
            else:
                if state.checkpoint_path is None:
                    raise ValueError(
                        "Local SAM scan requires --checkpoint or ROADLENS_SAM_CHECKPOINT"
                    )
                args = Namespace(
                    command="scan",
                    video=clip_path,
                    prompt=prompt,
                    checkpoint=state.checkpoint_path,
                    start=start_s,
                    duration=dur_s,
                    sampling="distance",
                    spacing=1.0,
                    max_interval=0.5,
                    sharpness_window=0.05,
                    width=0,
                    chunk_frames=64,
                    overlap_frames=16,
                    grounding_batch_size=4,
                    confidence=0.25,
                    detection_threshold=0.25,
                    new_object_threshold=0.35,
                    refine_padding=2.0,
                    no_refine=(sam_mode == "scan_coarse"),
                    resume=False,
                    plan_only=False,
                    output=pass_dir,
                )
                run_scan(args)

            if len(passes) > 1:
                # Re-index multi-pass observations into job_dir cleanly.
                sub_obs = json.loads((pass_dir / "observations.json").read_text())
                for item in sub_obs:
                    new_id = len(merged_obs)
                    old_fidx = int(item["best_frame_index"])
                    for prefix in ("overlay", "frame", "result"):
                        ext = "json" if prefix == "result" else "png"
                        src_file = pass_dir / f"{prefix}-{old_fidx:06d}.{ext}"
                        if src_file.is_file():
                            (job_dir / f"{prefix}-{new_id:06d}.{ext}").write_bytes(
                                src_file.read_bytes()
                            )
                    from datetime import datetime

                    merged_obs.append(
                        Observation(
                            object_id=new_id,
                            prompt=item["prompt"],
                            backend=item["backend"],
                            first_source_seconds=item["first_source_seconds"],
                            last_source_seconds=item["last_source_seconds"],
                            best_source_seconds=item["best_source_seconds"],
                            best_frame_index=new_id,
                            recorded_utc=datetime.fromisoformat(item["recorded_utc"]),
                            score=item["score"],
                            foreground_pixels=item["foreground_pixels"],
                            latitude=item["latitude"],
                            longitude=item["longitude"],
                            location_source=item["location_source"],
                        )
                    )

        if len(passes) > 1:
            write_observations(merged_obs, Path(passes[0]["clip_path"]), job_dir)

        # Free SAM GPU tensors before launching local Muse Glimmer on single 24 GB GPU!
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        street = street_names[0] if street_names else None
        place = place_names[0] if place_names else None
        if vlm_mode in ("local", "remote"):
            state.jobs[job_id]["progress"] = f"Running {vlm_mode} VLM triage & report..."
            with create_vlm_backend(vlm_mode) as vlm_backend:
                triage_run_directory(
                    job_dir,
                    vlm_backend,
                    street_name=street,
                    place_name=place,
                )
        else:
            obs_items = json.loads((job_dir / "observations.json").read_text())
            (job_dir / "summary-report.md").write_text(
                format_deterministic_summary_report(
                    obs_items, street_name=street, place_name=place
                ),
                encoding="utf-8",
            )

        run_id = state.register_run(job_dir)
        state.jobs[job_id].update(
            status="complete",
            progress="Scan complete — awaiting human review",
            run=state.serialize_run(run_id),
        )
    except Exception as err:
        state.jobs[job_id].update(
            status="failed",
            progress=f"{type(err).__name__}: {err}",
        )


class StudioHandler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, state: StudioState, **kwargs: Any):
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/__roadlens_osm.pmtiles":
            self._serve_pmtiles(send_body=False)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/":
            html = render_studio_html(self.state.initial_ride_path).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if route == "/__roadlens_osm.pmtiles":
            self._serve_pmtiles(send_body=True)
            return
        if route == "/api/presets":
            self._send_json({"presets": list(HAZARD_PRESETS)})
            return
        if route.startswith("/api/jobs/"):
            job_id = route.rsplit("/", 1)[-1]
            job = self.state.jobs.get(job_id)
            if not job:
                self._send_json({"error": "Unknown job"}, status=404)
            else:
                self._send_json(job)
            return
        if route.startswith("/runs/"):
            parts = route.strip("/").split("/")
            if len(parts) == 4 and parts[0] == "runs" and parts[2] == "files":
                run_id = parts[1]
                filename = unquote(parts[3])
                run_dir = self.state.runs.get(run_id)
                if not run_dir or "/" in filename or ".." in filename:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                target = run_dir / filename
                if not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                data = target.read_bytes()
                ctype = (
                    "image/png"
                    if filename.endswith(".png")
                    else "image/jpeg"
                    if filename.endswith(".jpg")
                    else "application/octet-stream"
                )
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/api/upload-clip":
                raw_name = unquote(self.headers.get("X-Filename") or "clip.mp4")
                safe_name = Path(raw_name).name or "clip.mp4"
                remaining = int(self.headers.get("Content-Length") or 0)
                if remaining <= 0:
                    raise ValueError("Empty video upload")
                uploads_dir = self.state.workspace_dir / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                dest = uploads_dir / safe_name
                with dest.open("wb") as out:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        out.write(chunk)
                        remaining -= len(chunk)
                self.state.ride = index_ride(dest)
                self._send_json(
                    {
                        "saved_path": str(dest),
                        "ride": self.state.ride.to_dict(),
                        "geojson": self.state.ride.to_geojson(),
                    }
                )
                return

            body = self._read_json()
            if route == "/api/index-ride":
                target = Path(str(body.get("path", "")).strip())
                self.state.ride = index_ride(target)
                self._send_json(
                    {
                        "ride": self.state.ride.to_dict(),
                        "geojson": self.state.ride.to_geojson(),
                    }
                )
                return

            if route == "/api/resolve":
                if self.state.ride is None:
                    raise ValueError("Index a ride clip or folder first")
                mode = str(body.get("mode") or "")
                if mode == "corridor":
                    raw_line = body.get("polyline") or []
                    line = [(float(pt[0]), float(pt[1])) for pt in raw_line]
                    passes = resolve_by_corridor(
                        self.state.ride,
                        polylines=[line],
                        tolerance_m=float(body.get("tolerance_m", 45.0)),
                        geocoder=self.state.street_index,
                    )
                elif mode == "street":
                    passes = resolve_by_street_name(
                        self.state.ride,
                        str(body.get("value", "")),
                        geocoder=self.state.street_index,
                    )
                elif mode == "offset":
                    passes = resolve_by_ride_offset(
                        self.state.ride,
                        body.get("value", 0),
                        window_seconds=float(body.get("window_seconds", 30.0)),
                        geocoder=self.state.street_index,
                    )
                elif mode == "clock":
                    passes = resolve_by_wall_clock(
                        self.state.ride,
                        str(body.get("value", "")),
                        window_seconds=float(body.get("window_seconds", 60.0)),
                        geocoder=self.state.street_index,
                    )
                else:
                    raise ValueError(f"Unsupported resolve mode: {mode!r}")
                self._send_json({"passes": [p.to_dict() for p in passes]})
                return

            if route == "/api/open-run":
                run_dir = Path(str(body.get("path", "")).strip())
                run_id = self.state.register_run(run_dir)
                self._send_json(self.state.serialize_run(run_id))
                return

            if route == "/api/jobs":
                passes = list(body.get("passes") or [])
                if not passes:
                    raise ValueError("At least one resolved pass is required")
                prompt = str(body.get("prompt") or "pothole").strip()
                sam_mode = str(body.get("sam_mode") or "scan_local")
                vlm_mode = str(body.get("vlm_mode") or "local")
                job_id = uuid.uuid4().hex[:10]
                self.state.jobs[job_id] = {
                    "job_id": job_id,
                    "status": "running",
                    "progress": "Queued",
                    "run": None,
                }
                worker = threading.Thread(
                    target=_execute_targeted_job,
                    args=(self.state, job_id, passes, prompt, sam_mode, vlm_mode),
                    daemon=True,
                )
                worker.start()
                self._send_json({"job_id": job_id, "status": "running"}, status=202)
                return

            if route.startswith("/api/runs/"):
                parts = route.strip("/").split("/")
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs":
                    run_id, action = parts[2], parts[3]
                    if run_id not in self.state.runs:
                        self._send_json({"error": "Unknown run_id"}, status=404)
                        return
                    run_dir = self.state.runs[run_id]
                    if action == "review":
                        status = self.state.update_review_decision(
                            run_id,
                            int(body["object_id"]),
                            str(body["review_status"]),
                        )
                        self._send_json({"review_status": status})
                        return
                    if action == "triage":
                        vlm_mode = str(body.get("vlm_mode") or "local")
                        with create_vlm_backend(vlm_mode) as vlm_backend:
                            triage_run_directory(
                                run_dir,
                                vlm_backend,
                                max_candidates=body.get("max_candidates"),
                            )
                        self._send_json(self.state.serialize_run(run_id))
                        return
                    if action == "draft-report":
                        vlm_mode = str(body.get("vlm_mode") or "none")
                        serialized = self.state.serialize_run(run_id)
                        obs_items = serialized["observations"]
                        if vlm_mode in ("local", "remote"):
                            with create_vlm_backend(vlm_mode) as vlm_backend:
                                md = vlm_backend.draft_summary_report(obs_items)
                        else:
                            md = format_deterministic_summary_report(obs_items)
                        (run_dir / "summary-report.md").write_text(md, encoding="utf-8")
                        self._send_json({"summary_report": md})
                        return
                    if action == "save-report":
                        md = str(body.get("summary_report") or "")
                        (run_dir / "summary-report.md").write_text(md, encoding="utf-8")
                        self._send_json({"saved": True})
                        return
            self._send_json({"error": "Not found"}, status=404)
        except Exception as err:
            self._send_json({"error": str(err)}, status=400)

    def _serve_pmtiles(self, send_body: bool) -> None:
        size = self.state.pmtiles_path.stat().st_size
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
        with self.state.pmtiles_path.open("rb") as archive:
            archive.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = archive.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def create_studio_server(
    workspace_dir: Path,
    pmtiles_path: Path,
    host: str = "127.0.0.1",
    port: int = 8766,
    checkpoint_path: Path | None = None,
    initial_ride_path: Path | None = None,
) -> ThreadingHTTPServer:
    state = StudioState(
        workspace_dir=workspace_dir,
        pmtiles_path=pmtiles_path,
        checkpoint_path=checkpoint_path or (
            Path(os.environ["ROADLENS_SAM_CHECKPOINT"])
            if os.environ.get("ROADLENS_SAM_CHECKPOINT")
            else None
        ),
        initial_ride_path=initial_ride_path,
    )
    handler = partial(StudioHandler, state=state)
    return ThreadingHTTPServer((host, port), handler)


def serve_studio(
    workspace_dir: Path,
    pmtiles_path: Path,
    host: str = "127.0.0.1",
    port: int = 8766,
    checkpoint_path: Path | None = None,
    initial_ride_path: Path | None = None,
) -> None:
    server = create_studio_server(
        workspace_dir=workspace_dir,
        pmtiles_path=pmtiles_path,
        host=host,
        port=port,
        checkpoint_path=checkpoint_path,
        initial_ride_path=initial_ride_path,
    )
    print(f"RoadLens Studio: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
