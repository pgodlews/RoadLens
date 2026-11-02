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
    _detection_box_for_observation,
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
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
  <style>
    :root {
      color-scheme: dark;
      --bg: #0E1012; --panel: #15181B; --raised: #1C2024; --raised-2: #1F2428; --line: #2A3036; --line-2: #353C43;
      --text: #ECEEF0; --text-2: #C9D0D6; --muted: #9AA3AB; --faint: #8A939B;
      --amber: #FFB020; --amber-ink: #1A1300; --amber-wash: #2A2312;
      --blue: #5AB8FF; --blue-ink: #04121D; --grey: #8A939B;
      --display: 'Barlow Condensed', 'Arial Narrow', sans-serif;
      --mono: 'JetBrains Mono', ui-monospace, monospace;
      font-family: Inter, system-ui, -apple-system, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-size: 13px; }
    button, input, select, textarea { font: inherit; color: inherit; }
    header { height: 64px; display: flex; align-items: center; gap: 24px; padding: 0 22px; border-bottom: 1px solid var(--line); background: #121417; }
    .brand { display: flex; align-items: center; gap: 10px; }
    .wordmark { font-family: var(--display); font-weight: 700; font-size: 26px; letter-spacing: 0.06em; text-transform: uppercase; }
    .wordmark span { font-weight: 500; color: var(--muted); margin-left: 6px; }
    .stepper { display: flex; align-items: center; gap: 4px; flex: 1 1 auto; }
    .step { padding: 6px 10px; border-radius: 6px; color: var(--faint); }
    .step.done { color: var(--text); }
    .step.current { background: var(--amber); color: var(--amber-ink); font-weight: 600; }
    .step-sep { color: #4A525A; }
    .ride-status { display: flex; align-items: center; gap: 8px; font-family: var(--mono); font-size: 12px; color: var(--muted); white-space: nowrap; }
    .ride-status .dot { width: 8px; height: 8px; border-radius: 50%; background: #4A525A; }
    .ride-status.ready .dot { background: var(--blue); }
    main { display: grid; grid-template-columns: 400px 1fr; height: calc(100vh - 64px); }
    aside { overflow-y: auto; padding: 20px; background: var(--panel); border-right: 1px solid var(--line); display: flex; flex-direction: column; gap: 24px; }
    section { display: flex; flex-direction: column; gap: 10px; }
    .sec-head { display: flex; align-items: baseline; gap: 10px; }
    .sec-num { font-family: var(--mono); font-size: 12px; color: var(--amber); }
    h2 { margin: 0; font-family: var(--display); font-weight: 600; font-size: 20px; letter-spacing: 0.05em; text-transform: uppercase; }
    .muted { color: var(--muted); font-size: 12px; line-height: 1.5; }
    .mono { font-family: var(--mono); }
    label.field { display: block; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin: 4px 0 2px; }
    input[type="text"], select, textarea {
      width: 100%; height: 40px; background: var(--bg); border: 1px solid var(--line); border-radius: 9px; padding: 0 12px; font-size: 13px;
    }
    textarea { height: auto; min-height: 180px; padding: 10px 12px; resize: vertical; line-height: 1.5; font-family: var(--mono); font-size: 12px; }
    input:focus, select:focus, textarea:focus, button:focus-visible { outline: 2px solid var(--amber); outline-offset: 1px; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }
    .row > .fit { flex: 0 0 auto; }
    button { height: 40px; padding: 0 14px; border: 1px solid var(--line-2); border-radius: 9px; background: transparent; cursor: pointer; font-weight: 500; }
    button:hover:not(:disabled) { background: var(--raised); }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    button.primary { background: var(--amber); border-color: var(--amber); color: var(--amber-ink); font-weight: 600; }
    button.primary:hover:not(:disabled) { background: #FFC24D; }
    button.cta { height: 52px; border-radius: 12px; font-family: var(--display); font-weight: 700; font-size: 20px; letter-spacing: 0.06em; text-transform: uppercase; display: flex; align-items: center; justify-content: center; gap: 10px; }
    button.confirm-btn { background: var(--blue); border-color: var(--blue); color: var(--blue-ink); font-weight: 600; }
    button.confirm-btn:hover:not(:disabled) { background: #7CC6FF; }
    .card { padding: 12px; border-radius: 10px; background: var(--raised); border: 1px solid var(--line); }
    .segmented { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 4px; padding: 4px; border-radius: 10px; background: var(--bg); border: 1px solid var(--line); }
    .segmented button { height: 34px; border: 0; border-radius: 7px; color: var(--muted); }
    .segmented button.active { background: var(--line); color: var(--text); font-weight: 600; }
    .dropzone { border: 1px dashed var(--line-2); border-radius: 10px; padding: 12px; text-align: center; font-size: 12px; color: var(--muted); cursor: pointer; }
    .dropzone.drag { border-color: var(--amber); background: var(--amber-wash); color: var(--text); }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip { height: 34px; padding: 0 12px; border-radius: 17px; border: 1px solid var(--line-2); color: var(--text-2); }
    .chip.active { background: var(--amber); border-color: var(--amber); color: var(--amber-ink); font-weight: 600; }
    .pass-item { display: flex; gap: 10px; align-items: flex-start; padding: 12px; border-radius: 10px; background: var(--raised); border: 1px solid var(--line); cursor: pointer; }
    .pass-item:has(input:checked) { border-color: var(--amber); }
    .pass-item input { margin-top: 2px; accent-color: var(--amber); width: 16px; height: 16px; }
    .pass-title { font-weight: 600; }
    .pass-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin-top: 8px; font-family: var(--mono); font-size: 12px; }
    .pass-stats small { display: block; font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .heading-pill { font-family: var(--mono); font-size: 11px; padding: 2px 7px; border-radius: 5px; background: var(--amber-wash); color: var(--amber); margin-left: 6px; }
    .engines { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .warn { background: #2A2312; border: 1px solid #6E5227; color: #F2CF8D; padding: 10px 12px; border-radius: 9px; font-size: 12px; line-height: 1.45; }
    .guard { display: flex; gap: 8px; align-items: flex-start; font-size: 12px; color: var(--muted); line-height: 1.45; }
    .tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
    .tile { padding: 8px 10px; border-radius: 8px; background: var(--raised); display: flex; flex-direction: column; gap: 2px; }
    .tile b { font-family: var(--mono); font-size: 20px; font-weight: 500; }
    .tile span { font-size: 11px; color: var(--muted); }
    .queue { display: flex; flex-direction: column; gap: 2px; }
    .q-row { display: flex; align-items: center; gap: 12px; width: 100%; height: auto; padding: 8px; border-radius: 10px; border: 1px solid transparent; text-align: left; }
    .q-row.selected { background: var(--raised-2); border-color: var(--amber); }
    .q-row img, .q-row .noimg { width: 52px; height: 52px; border-radius: 8px; object-fit: cover; flex: 0 0 auto; background: var(--raised); }
    .q-row.rejected img { opacity: 0.45; }
    .q-body { display: flex; flex-direction: column; gap: 5px; flex: 1 1 auto; min-width: 0; }
    .q-top { display: flex; justify-content: space-between; gap: 6px; font-family: var(--mono); font-size: 12px; }
    .q-top .score { color: var(--muted); }
    .q-bar { height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; }
    .q-bar span { display: block; height: 100%; background: var(--text-2); }
    .q-meta { display: flex; align-items: center; gap: 6px; font-size: 12px; min-width: 0; }
    .q-meta .vlm { color: var(--faint); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sdot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; }
    .s-unreviewed { color: var(--amber); } .sdot.s-unreviewed { border: 2px solid var(--amber); }
    .s-confirmed { color: var(--blue); } .sdot.s-confirmed { background: var(--blue); }
    .s-rejected { color: var(--grey); } .sdot.s-rejected { background: var(--grey); }
    .stage { position: relative; min-width: 0; }
    #map { position: absolute; inset: 0; background: #111417; }
    .leaflet-container { font-family: Inter, system-ui, sans-serif; background: #111417; }
    .leaflet-bar a, .leaflet-bar a:hover { background: var(--panel); color: var(--text); border-color: var(--line); width: 40px; height: 40px; line-height: 40px; }
    .leaflet-control-scale-line { background: rgba(21, 24, 27, 0.8); color: var(--muted); border-color: var(--muted); font-family: var(--mono); }
    .leaflet-control-attribution { background: rgba(14, 16, 18, 0.8) !important; color: var(--faint); }
    .leaflet-control-attribution a { color: var(--muted); }
    .leaflet-tooltip { background: var(--panel); color: var(--text); border-color: var(--line); font-family: var(--mono); }
    .float { position: absolute; z-index: 500; background: rgba(21, 24, 27, 0.96); border: 1px solid var(--line); border-radius: 14px; }
    #jobCard { left: 20px; bottom: 20px; width: 380px; max-width: calc(100% - 40px); padding: 16px; display: none; flex-direction: column; gap: 12px; }
    #jobCard.show { display: flex; }
    .job-head { display: flex; justify-content: space-between; align-items: baseline; }
    .job-head strong { font-family: var(--display); font-weight: 600; font-size: 18px; letter-spacing: 0.05em; text-transform: uppercase; }
    .job-bar { height: 6px; border-radius: 3px; background: var(--line); overflow: hidden; }
    .job-bar span { display: block; height: 100%; width: 0; background: var(--amber); transition: width 0.4s; }
    .job-steps { display: flex; flex-direction: column; gap: 8px; }
    .job-step { display: flex; align-items: center; gap: 10px; color: var(--faint); }
    .job-step i { width: 10px; height: 10px; border-radius: 50%; border: 1.5px solid #4A525A; flex: 0 0 auto; }
    .job-step.done { color: var(--text); } .job-step.done i { background: var(--blue); border-color: var(--blue); }
    .job-step.active { color: var(--text); } .job-step.active i { background: var(--amber); border-color: var(--amber); box-shadow: 0 0 0 4px rgba(255, 176, 32, 0.18); }
    #mapHint { left: 20px; top: 20px; padding: 10px 14px; font-size: 12px; color: var(--muted); }
    #reviewPane { position: absolute; inset: 0; z-index: 1100; background: var(--bg); display: none; grid-template-columns: minmax(0, 1fr) 360px; }
    #reviewPane.open { display: grid; }
    .viewer { padding: 20px; display: flex; flex-direction: column; gap: 14px; overflow: auto; min-width: 0; }
    .viewer-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; }
    .viewer-head h1 { margin: 0; font-family: var(--display); font-weight: 700; font-size: 34px; letter-spacing: 0.03em; }
    .status-tag { font-family: var(--mono); font-size: 12px; padding: 4px 8px; border-radius: 6px; margin-left: 12px; text-transform: uppercase; vertical-align: middle; }
    .status-tag.s-unreviewed { background: var(--amber-wash); }
    .status-tag.s-confirmed { background: #0F2433; }
    .status-tag.s-rejected { background: var(--raised); }
    .evidence { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 16px; align-items: start; }
    @media (max-width: 1180px) { .evidence { grid-template-columns: 1fr; } }
    .frame-wrap { position: relative; min-width: 0; border-radius: 12px; overflow: hidden; background: var(--panel); }
    .frame-wrap img { display: block; width: 100%; height: auto; }
    .bbox { position: absolute; border: 2px solid var(--amber); border-radius: 4px; box-shadow: 0 0 0 9999px rgba(14, 16, 18, 0.35); pointer-events: none; }
    .bbox-label { position: absolute; transform: translateY(-100%); margin-top: -6px; font-family: var(--mono); font-size: 12px; padding: 4px 8px; border-radius: 6px; background: var(--amber); color: var(--amber-ink); white-space: nowrap; }
    .crop-wrap { min-width: 0; display: flex; flex-direction: column; gap: 8px; }
    .crop { width: 100%; aspect-ratio: 1 / 1; border-radius: 12px; background-color: var(--panel); background-repeat: no-repeat; }
    figcaption { display: flex; justify-content: space-between; gap: 8px; font-family: var(--mono); font-size: 12px; color: var(--muted); }
    .crop-wrap figcaption { font-family: inherit; line-height: 1.5; }
    .decide { padding: 20px; background: var(--panel); border-left: 1px solid var(--line); display: flex; flex-direction: column; gap: 14px; overflow: auto; }
    .decide .card { display: flex; flex-direction: column; gap: 8px; padding: 16px; border-radius: 12px; }
    .kicker { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .verdict { font-family: var(--display); font-weight: 600; font-size: 24px; color: var(--amber); }
    .verdict.v-calm { color: var(--text-2); }
    .quote { margin: 0; font-size: 13px; line-height: 1.55; color: var(--text-2); }
    .quote.clamp { display: -webkit-box; -webkit-line-clamp: 7; -webkit-box-orient: vertical; overflow: hidden; cursor: pointer; }
    .decide-actions { display: flex; flex-direction: column; gap: 8px; margin-top: auto; }
    .decide-actions .confirm-btn { height: 56px; border-radius: 12px; font-family: var(--display); font-weight: 700; font-size: 22px; letter-spacing: 0.05em; text-transform: uppercase; display: flex; align-items: center; justify-content: center; gap: 12px; }
    .decide-actions .pair { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .decide-actions .pair button { height: 46px; border-radius: 12px; display: flex; align-items: center; justify-content: center; gap: 6px; }
    kbd { font-family: var(--mono); font-size: 11px; padding: 1px 6px; border-radius: 5px; background: rgba(255, 255, 255, 0.08); color: var(--muted); }
    .confirm-btn kbd { background: rgba(4, 18, 29, 0.15); color: var(--blue-ink); }
    .view-toggle { display: flex; gap: 4px; padding: 4px; border-radius: 10px; background: var(--panel); border: 1px solid var(--line); }
    .view-toggle button { height: 34px; border: 0; border-radius: 7px; color: var(--muted); }
    .view-toggle button.active { background: var(--line); color: var(--text); font-weight: 600; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; grid-template-rows: auto 70vh; height: auto; }
      aside { order: 2; }
      #reviewPane.open { grid-template-columns: 1fr; position: fixed; overflow: auto; }
      header { flex-wrap: wrap; height: auto; padding: 12px 16px; }
    }
  </style>
</head>
<body>
<header>
  <div class="brand" title="Bookended Human-in-the-Loop · Scope on OSM -> Targeted Scan/VLM -> Confirm & Export">
    <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true"><circle cx="14" cy="14" r="12" fill="none" stroke="#FFB020" stroke-width="2.5"/><path d="M10.5 24 L13 7 M17.5 24 L15 7" stroke="#ECEEF0" stroke-width="2" stroke-linecap="round" fill="none"/><path d="M14 21 v-2.5 M14 15.5 v-2.5 M14 10.5 v-1.5" stroke="#FFB020" stroke-width="1.6" fill="none"/></svg>
    <span class="wordmark">RoadLens<span>Studio</span></span>
  </div>
  <nav class="stepper" aria-label="Workflow" id="stepper"></nav>
  <span class="ride-status" id="rideStatusWrap"><span class="dot"></span><span id="rideStatus">No ride indexed yet</span></span>
</header>
<main>
  <aside>
    <section>
      <div class="sec-head"><span class="sec-num">01</span><h2>Ride</h2></div>
      <label class="field" for="ridePath">Garmin clip (.MP4) or ride folder</label>
      <div class="row">
        <input id="ridePath" type="text" placeholder="/path/to/GRMS0010.MP4 or folder" value="__INITIAL_RIDE_PATH__">
        <button id="loadRideBtn" class="primary fit">Index GPS</button>
      </div>
      <div id="dropzone" class="dropzone" role="button" tabindex="0">
        Drop an .MP4 clip here, or click to import<input id="fileInput" type="file" accept=".mp4,.MP4" style="display:none">
      </div>
      <label class="field" for="existingRunPath">Or reopen a scan folder</label>
      <div class="row">
        <input id="existingRunPath" type="text" placeholder="artifacts/eccup-survey/return">
        <button id="openRunBtn" class="fit">Open run</button>
      </div>
    </section>

    <section>
      <div class="sec-head"><span class="sec-num">02</span><h2>Scope a stretch</h2></div>
      <div class="muted">Click two points on the ride track to select a stretch (outbound and return passes are both found), or search:</div>
      <div class="segmented" role="group" aria-label="Search by">
        <button type="button" data-mode="street" class="active">Street</button>
        <button type="button" data-mode="offset">Ride time</button>
        <button type="button" data-mode="clock">Clock</button>
      </div>
      <input id="queryMode" type="hidden" value="street">
      <div class="row">
        <input id="queryValue" type="text" aria-label="Search value" placeholder="Eccup Moor Road">
        <button id="resolveQueryBtn" class="fit">Find</button>
      </div>
      <div id="resolvedPasses" style="display:flex; flex-direction:column; gap:8px"></div>
      <button id="clearMapSelBtn" type="button">Clear selection</button>
    </section>

    <section>
      <div class="sec-head"><span class="sec-num">03</span><h2>Look for</h2></div>
      <div id="presetChips" class="chips"></div>
      <input id="hazardPrompt" type="text" value="pothole" aria-label="Hazard description" placeholder="Describe a hazard, e.g. fallen tree branch">
      <div class="engines">
        <div>
          <label class="field" for="samMode">Segment · SAM 3.1</label>
          <select id="samMode">
            <option value="scan_local">Local · 1 m + 30 fps refine</option>
            <option value="scan_coarse">Local · 1 m coarse only</option>
            <option value="analyze_remote">Remote Meta API</option>
          </select>
        </div>
        <div>
          <label class="field" for="vlmMode">Triage · VLM</label>
          <select id="vlmMode">
            <option value="local">Muse Glimmer (local)</option>
            <option value="none">Skip VLM (template)</option>
            <option value="remote">Remote API VLM</option>
          </select>
        </div>
      </div>
      <div id="remotePrivacyWarn" class="warn" style="display:none">
        Remote mode sends selected frames to the configured API endpoint. The burned-in Garmin GPS bar is masked on remote VLM calls.
      </div>
      <button id="runScanBtn" class="primary cta" disabled>
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M5 3 L15 9 L5 15 Z" fill="#1A1300"/></svg>
        Scan selected pass
      </button>
      <div id="jobStatus" class="muted mono" aria-live="polite"></div>
    </section>

    <section>
      <div class="sec-head"><span class="sec-num">04</span><h2>Review &amp; export</h2></div>
      <div class="muted" id="reviewSummary">Run or open a scan to review candidates. Unreviewed model candidates are never exported as confirmed hazards.</div>
      <div class="tiles" id="reviewTiles" style="display:none">
        <div class="tile"><b class="s-unreviewed" id="cntUnreviewed">0</b><span>unreviewed</span></div>
        <div class="tile"><b class="s-confirmed" id="cntConfirmed">0</b><span>confirmed</span></div>
        <div class="tile"><b class="s-rejected" id="cntRejected">0</b><span>rejected</span></div>
      </div>
      <div id="candidateCards" class="queue"></div>
      <div class="row">
        <button id="triageRunBtn" disabled>Run VLM triage</button>
        <button id="draftReportBtn" disabled>Draft report</button>
      </div>
      <div id="reportSection" style="display:none; flex-direction:column; gap:8px">
        <label class="field" for="reportEditor">Summary report draft · check before any manual council / FixMyStreet submission</label>
        <textarea id="reportEditor"></textarea>
        <div class="row">
          <button id="saveReportBtn">Save draft</button>
          <button id="downloadReportBtn">Download .md</button>
        </div>
      </div>
      <div class="row">
        <button id="exportConfirmedBtn" class="confirm-btn" disabled>Export confirmed (0)</button>
        <button id="exportAuditBtn" class="fit" disabled>Full audit</button>
      </div>
    </section>

    <div class="guard">
      <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" style="flex:0 0 auto; margin-top:1px"><path d="M8 1.5 L13.5 3.5 V8 C13.5 11 11 13.5 8 14.5 C5 13.5 2.5 11 2.5 8 V3.5 Z" fill="none" stroke="#9AA3AB" stroke-width="1.4"/></svg>
      <span>Bookended Human-in-the-Loop: you scope the stretch and you confirm every hazard. Nothing is submitted automatically.</span>
    </div>
  </aside>

  <div class="stage">
    <div id="map"></div>
    <div id="mapHint" class="float">Index a ride, then click two points on its track.</div>
    <div id="jobCard" class="float" aria-live="polite">
      <div class="job-head"><strong id="jobTitle">Scanning</strong><span class="mono muted" id="jobClock">0:00</span></div>
      <div class="job-bar"><span id="jobBar"></span></div>
      <div class="job-steps" id="jobSteps"></div>
    </div>

    <div id="reviewPane" role="dialog" aria-label="Candidate review">
      <div class="viewer">
        <div class="viewer-head">
          <div><h1 style="display:inline" id="rvTitle">Candidate</h1><span class="status-tag" id="rvStatus"></span></div>
          <div class="row" style="flex:0 0 auto">
            <div class="view-toggle" role="group" aria-label="Evidence view">
              <button type="button" data-view="box" class="active">Box</button>
              <button type="button" data-view="mask">SAM mask</button>
              <button type="button" data-view="raw">Raw</button>
            </div>
            <button type="button" id="rvClose" class="fit">Map <kbd>Esc</kbd></button>
          </div>
        </div>
        <div class="evidence">
          <figure style="margin:0; min-width:0; display:flex; flex-direction:column; gap:8px">
            <div class="frame-wrap" id="rvFrameWrap">
              <img id="rvFrame" alt="Camera frame for the selected candidate">
              <div class="bbox" id="rvBox"></div>
              <span class="bbox-label" id="rvBoxLabel"></span>
            </div>
            <figcaption><span id="rvFrameCaption"></span><span id="rvFrameSize"></span></figcaption>
          </figure>
          <figure class="crop-wrap" style="margin:0">
            <div class="crop" id="rvCrop" role="img" aria-label="Unmasked close-up of the candidate"></div>
            <figcaption>Unmasked 1:1 close-up: the view the VLM triages, so the mask never hides the surface texture.</figcaption>
          </figure>
        </div>
      </div>
      <div class="decide">
        <div class="card" id="rvTriage"></div>
        <div class="card">
          <span class="kicker">Camera GPS</span>
          <span class="mono" style="font-size:15px" id="rvGps"></span>
          <span class="mono muted" id="rvWhen"></span>
          <span class="muted">Where the bike was when the frame was captured, not a surveyed defect position.</span>
        </div>
        <div class="decide-actions">
          <button type="button" class="confirm-btn" id="rvConfirm">
            <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10.5 l4 4 L16 6" stroke="#04121D" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
            Confirm hazard <kbd>C</kbd>
          </button>
          <div class="pair">
            <button type="button" id="rvPrev">Prev <kbd>&larr;</kbd></button>
            <button type="button" id="rvReject">Reject <kbd>X</kbd></button>
            <button type="button" id="rvNext">Next <kbd>&rarr;</kbd></button>
          </div>
        </div>
      </div>
    </div>
  </div>
</main>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/pmtiles@4.3.0/dist/pmtiles.js"></script>
<script src="https://unpkg.com/protomaps-leaflet@5.0.0/dist/protomaps-leaflet.js"></script>
<script>
const PRESETS = __HAZARD_PRESETS__;
const AMBER = '#FFB020', BLUE = '#5AB8FF', GREY = '#8A939B', TRACK = '#C9D2DA';
const map = L.map('map', {zoomControl: false}).setView([53.85, -1.54], 12);
L.control.zoom({position: 'bottomright'}).addTo(map);
L.control.scale({position: 'bottomright', imperial: false}).addTo(map);
try {
  if (window.protomapsL && protomapsL.leafletLayer) {
    protomapsL.leafletLayer({url: '/__roadlens_osm.pmtiles', flavor: 'dark', attribution: '© OpenStreetMap contributors · offline PMTiles'}).addTo(map);
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
let reviewIndex = -1;
let evidenceView = 'box';

function esc(val) {
  return String(val ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const STEPS = ['Scope', 'Scan', 'Triage', 'Review', 'Export'];
function setStep(current) {
  const idx = STEPS.indexOf(current);
  document.getElementById('stepper').innerHTML = STEPS.map((s, i) =>
    '<span class="step ' + (i === idx ? 'current' : i < idx ? 'done' : '') + '">' + s + '</span>' +
    (i < STEPS.length - 1 ? '<span class="step-sep" aria-hidden="true">›</span>' : '')
  ).join('');
}
setStep('Scope');

function renderPresets() {
  const wrap = document.getElementById('presetChips');
  const input = document.getElementById('hazardPrompt');
  PRESETS.forEach((p, idx) => {
    const el = document.createElement('button');
    el.type = 'button';
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

const QUERY_HINTS = {street: 'Eccup Moor Road', offset: '20m or 1h05m', clock: '12:30'};
document.querySelectorAll('.segmented button').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.segmented button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('queryMode').value = btn.dataset.mode;
    document.getElementById('queryValue').placeholder = QUERY_HINTS[btn.dataset.mode];
  };
});

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
  currentPasses = [];
  clickLayer.clearLayers();
  document.getElementById('resolvedPasses').innerHTML = '';
  document.getElementById('runScanBtn').disabled = true;
  const summary = ridePayload.ride;
  document.getElementById('rideStatus').textContent =
    summary.primary_clip_count + ' clips · ' +
    (summary.total_distance_m / 1000).toFixed(1) + ' km · ' +
    Math.round(summary.total_elapsed_seconds / 60) + ' min';
  document.getElementById('rideStatusWrap').classList.add('ready');
  document.getElementById('mapHint').textContent = 'Click two points on the track to scope a stretch.';
  const geo = L.geoJSON(ridePayload.geojson, {
    style: {color: TRACK, weight: 3, opacity: 0.55}
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
  L.circleMarker(latlng, {radius: 8, weight: 3, color: '#ECEEF0', fillColor: '#0E1012', fillOpacity: 1}).addTo(clickLayer);
  if (mapClicks.length === 2) {
    L.polyline(mapClicks, {color: AMBER, dashArray: '4,6', weight: 2}).addTo(clickLayer);
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
      L.polyline(p.polyline, {color: AMBER, weight: 20, opacity: 0.2, lineCap: 'round'}).addTo(passLayer);
      const line = L.polyline(p.polyline, {color: AMBER, weight: 6, opacity: 1, lineCap: 'round'}).addTo(passLayer);
      allBounds.push(...p.polyline);
      line.bindPopup('Pass ' + esc(p.pass_index) + ' · ' + esc(p.clip_name));
    }
    const street = [p.street_name, p.place_name].filter(Boolean).join(' · ') || 'Unnamed road';
    return '<label class="pass-item">' +
      '<input type="checkbox" class="passCheck" data-idx="' + idx + '" checked>' +
      '<span style="flex:1; min-width:0">' +
      '<span class="pass-title">Pass ' + esc(p.pass_index) + ' · ' + esc(p.clip_name) + '</span>' +
      '<span class="heading-pill">' + esc(p.compass_direction || '–') + '</span>' +
      '<span class="pass-stats">' +
      '<span><small>window</small>' + p.start_seconds.toFixed(0) + '–' + p.end_seconds.toFixed(0) + ' s</span>' +
      '<span><small>distance</small>' + Math.round(p.distance_m) + ' m</span>' +
      '<span><small>frames</small>~' + esc(p.estimated_coarse_frames_1m) + '</span>' +
      '</span>' +
      '<span class="muted" style="display:block; margin-top:6px">' + esc(street) + '</span>' +
      '</span></label>';
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
dropzone.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } };
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
document.getElementById('queryValue').onkeydown = (e) => {
  if (e.key === 'Enter') document.getElementById('resolveQueryBtn').click();
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

const JOB_STAGES = [
  {key: 'prep', label: 'Sample a sharp frame every metre'},
  {key: 'sam', label: 'Segment with SAM 3.1'},
  {key: 'vlm', label: 'Triage with the VLM'},
  {key: 'done', label: 'Ready for your review'}
];
function jobStage(job) {
  const text = (job.progress || '').toLowerCase();
  if (job.status === 'complete') return 4;
  if (text.includes('vlm') || text.includes('triage')) return 2;
  if (text.includes('pass')) return 1;
  return 0;
}
let jobTimer = null;
function showJob(job, startedAt) {
  const card = document.getElementById('jobCard');
  card.classList.add('show');
  const stage = jobStage(job);
  const failed = job.status === 'failed';
  document.getElementById('jobTitle').textContent = failed ? 'Scan failed' : stage >= 4 ? 'Scan complete' : stage === 2 ? 'Triaging' : 'Scanning';
  document.getElementById('jobBar').style.width = Math.min(100, [8, 45, 80, 100, 100][stage]) + '%';
  document.getElementById('jobBar').style.background = failed ? '#C0524A' : stage >= 4 ? BLUE : AMBER;
  document.getElementById('jobSteps').innerHTML = JOB_STAGES.map((s, i) =>
    '<div class="job-step ' + (i < stage || stage >= 4 ? 'done' : i === stage ? 'active' : '') + '"><i></i><span>' + esc(s.label) + '</span></div>'
  ).join('') + (job.progress ? '<div class="muted mono">' + esc(job.progress) + '</div>' : '');
  clearInterval(jobTimer);
  if (job.status === 'running') {
    jobTimer = setInterval(() => {
      const s = Math.floor((Date.now() - startedAt) / 1000);
      document.getElementById('jobClock').textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
    }, 1000);
  }
}

document.getElementById('runScanBtn').onclick = async () => {
  const checks = [...document.querySelectorAll('.passCheck:checked')];
  const chosen = checks.map(c => currentPasses[Number(c.dataset.idx)]);
  if (!chosen.length) { alert('Select at least one pass.'); return; }
  const prompt = document.getElementById('hazardPrompt').value.trim() || 'pothole';
  const sam_mode = document.getElementById('samMode').value;
  const vlm_mode = document.getElementById('vlmMode').value;
  document.getElementById('runScanBtn').disabled = true;
  document.getElementById('jobStatus').textContent = 'Starting targeted job...';
  setStep('Scan');
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
  const startedAt = Date.now();
  const tick = async () => {
    const job = await api('/api/jobs/' + jobId);
    statusEl.textContent = '[' + job.status + '] ' + (job.progress || '');
    showJob(job, startedAt);
    if (jobStage(job) === 2) setStep('Triage');
    if (job.status === 'complete' && job.run) {
      document.getElementById('runScanBtn').disabled = false;
      renderRun(job.run);
      return;
    }
    if (job.status === 'failed') {
      statusEl.textContent = 'Failed: ' + (job.error || job.progress || 'see server log');
      document.getElementById('runScanBtn').disabled = false;
      return;
    }
    setTimeout(tick, 1500);
  };
  await tick();
}

function markerColour(status) {
  return status === 'confirmed' ? BLUE : status === 'rejected' ? GREY : AMBER;
}

function vlmLabel(triage) {
  if (!triage || !triage.classification) return 'not triaged';
  return String(triage.classification).replace(/_/g, ' ');
}

function updateCounts() {
  if (!activeRun) return;
  const obs = activeRun.observations;
  const confirmed = obs.filter(o => o.review_status === 'confirmed').length;
  const rejected = obs.filter(o => o.review_status === 'rejected').length;
  const unreviewed = obs.length - confirmed - rejected;
  document.getElementById('reviewSummary').innerHTML =
    '<b>' + obs.length + '</b> candidate(s) in <code>' + esc(activeRun.run_name) + '</code>, sorted by SAM score. Click one to review it.';
  document.getElementById('reviewTiles').style.display = obs.length ? 'grid' : 'none';
  document.getElementById('cntUnreviewed').textContent = unreviewed;
  document.getElementById('cntConfirmed').textContent = confirmed;
  document.getElementById('cntRejected').textContent = rejected;
  const btn = document.getElementById('exportConfirmedBtn');
  btn.textContent = 'Export confirmed (' + confirmed + ')';
  btn.disabled = confirmed === 0;
  document.getElementById('exportAuditBtn').disabled = obs.length === 0;
  document.getElementById('triageRunBtn').disabled = obs.length === 0;
  document.getElementById('draftReportBtn').disabled = false;
  setStep(confirmed > 0 ? 'Export' : 'Review');
}

function queueRowHtml(item, idx) {
  const score = item.score === null || item.score === undefined ? null : Number(item.score);
  const thumb = item.frame_url || item.overlay_url;
  const status = item.review_status;
  return '<button type="button" class="q-row ' + esc(status) + (idx === reviewIndex ? ' selected' : '') + '" data-idx="' + idx + '">' +
    (thumb ? '<span class="noimg" style="' + cropStyle(item, 52) + '" aria-hidden="true"></span>' : '<span class="noimg"></span>') +
    '<span class="q-body">' +
    '<span class="q-top"><span>#' + esc(item.object_id) + ' · ' + Number(item.best_source_seconds).toFixed(2) + ' s</span><span class="score">' + (score === null ? 'n/a' : score.toFixed(3)) + '</span></span>' +
    '<span class="q-bar"><span style="width:' + Math.round((score || 0) * 100) + '%"></span></span>' +
    '<span class="q-meta"><span class="sdot s-' + esc(status) + '"></span><span class="s-' + esc(status) + '">' + esc(status) + '</span><span class="vlm">· ' + esc(vlmLabel(item.vlm_triage)) + '</span></span>' +
    '</span></button>';
}

function sortedObservations(run) {
  return [...run.observations].sort((a, b) => (Number(b.score) || 0) - (Number(a.score) || 0));
}

function renderQueue() {
  const container = document.getElementById('candidateCards');
  container.innerHTML = activeRun.ordered.map(queueRowHtml).join('');
  container.querySelectorAll('.q-row').forEach(row => {
    row.onclick = () => openReview(Number(row.dataset.idx));
  });
}

function renderRun(run) {
  run.ordered = sortedObservations(run);
  activeRun = run;
  candidateLayer.clearLayers();
  candidateMarkers.clear();
  const bounds = [];

  if (run.summary_report) {
    document.getElementById('reportSection').style.display = 'flex';
    document.getElementById('reportEditor').value = run.summary_report;
  } else {
    document.getElementById('reportSection').style.display = 'none';
  }

  run.ordered.forEach((item, idx) => {
    if (item.latitude !== null && item.longitude !== null) {
      const pt = [item.latitude, item.longitude];
      bounds.push(pt);
      const m = L.circleMarker(pt, {
        radius: 8, weight: 2, color: '#0E1012',
        fillColor: markerColour(item.review_status), fillOpacity: 1
      }).addTo(candidateLayer);
      m.bindTooltip('#' + esc(item.object_id) + ' · ' + esc(item.review_status));
      m.on('click', (ev) => { L.DomEvent.stopPropagation(ev); openReview(idx); });
      candidateMarkers.set(item.object_id, m);
    }
  });
  renderQueue();
  if (bounds.length) map.fitBounds(bounds, {padding: [80, 80], maxZoom: 18});
  document.getElementById('mapHint').textContent = 'Click a candidate marker or a row in the list to review it.';
  updateCounts();
}

function frameAspect() {
  const img = document.getElementById('rvFrame');
  return img.naturalWidth && img.naturalHeight ? img.naturalWidth / img.naturalHeight : 2688 / 2160;
}

function cropWindow(item, aspect) {
  const box = item.box_xywh_normalized;
  if (!box) return null;
  const [x, y, w, h] = box.map(Number);
  // Square window around the box, in normalised units of each axis.
  let sidePxOverH = Math.min(1, Math.max(w * aspect, h, 0.16) * 2.0);
  const sw = Math.min(1, sidePxOverH / aspect), sh = sidePxOverH;
  const left = Math.min(Math.max(0, x + w / 2 - sw / 2), 1 - sw);
  const top = Math.min(Math.max(0, y + h / 2 - sh / 2), 1 - sh);
  return {sw, sh, left, top};
}

function cropStyle(item, sizePx) {
  const url = item.frame_url || item.overlay_url;
  const win = cropWindow(item, 2688 / 2160);
  if (!url) return '';
  if (!win) return 'background:url(' + esc(url) + ') center/cover';
  const px = win.sw >= 1 ? 0 : (win.left / (1 - win.sw)) * 100;
  const py = win.sh >= 1 ? 0 : (win.top / (1 - win.sh)) * 100;
  return 'background-image:url(' + esc(url) + ');background-size:' + (100 / win.sw) + '% ' + (100 / win.sh) + '%;background-position:' + px + '% ' + py + '%;background-repeat:no-repeat';
}

function renderTriage(item) {
  const t = item.vlm_triage;
  const el = document.getElementById('rvTriage');
  if (!t || !t.classification) {
    el.innerHTML = '<span class="kicker">Model triage</span><span class="verdict v-calm">Not triaged yet</span>' +
      '<span class="muted">Run VLM triage from the sidebar to get a second opinion. It never changes the review status.</span>';
    return;
  }
  const calm = t.classification !== 'hazard_candidate';
  el.innerHTML =
    '<div style="display:flex; justify-content:space-between"><span class="kicker">Model triage · ' + esc(t.model || 'VLM') + '</span><span class="mono muted">' + esc(t.confidence || '') + '</span></div>' +
    '<span class="verdict' + (calm ? ' v-calm' : '') + '">' + esc(vlmLabel(t)) + '</span>' +
    (t.hazard_type || t.lane_position ? '<span style="color:var(--text-2)">' + esc([t.hazard_type, t.lane_position].filter(Boolean).join(' · ')) + '</span>' : '') +
    (t.visual_description ? '<p class="quote clamp" title="Click to expand" onclick="this.classList.toggle(&quot;clamp&quot;)">' + esc(t.visual_description) + '</p>' : '') +
    (t.uncertainty_note ? '<p class="quote muted" style="font-size:12px">Uncertainty: ' + esc(t.uncertainty_note) + '</p>' : '');
}

function paintEvidence() {
  const item = activeRun.ordered[reviewIndex];
  const img = document.getElementById('rvFrame');
  const url = evidenceView === 'mask' ? (item.overlay_url || item.frame_url) : (item.frame_url || item.overlay_url);
  if (img.getAttribute('src') !== url) img.setAttribute('src', url || '');
  const box = item.box_xywh_normalized;
  const showBox = evidenceView === 'box' && box;
  const boxEl = document.getElementById('rvBox'), label = document.getElementById('rvBoxLabel');
  boxEl.style.display = label.style.display = showBox ? 'block' : 'none';
  if (showBox) {
    const [x, y, w, h] = box.map(Number);
    Object.assign(boxEl.style, {left: x * 100 + '%', top: y * 100 + '%', width: w * 100 + '%', height: h * 100 + '%'});
    Object.assign(label.style, {left: x * 100 + '%', top: y * 100 + '%'});
    label.textContent = 'SAM 3.1 · ' + (item.score === null ? 'n/a' : Number(item.score).toFixed(3));
  }
  const crop = document.getElementById('rvCrop');
  const win = cropWindow(item, frameAspect());
  const cropUrl = item.frame_url || item.overlay_url;
  if (cropUrl && win) {
    crop.style.backgroundImage = 'url(' + cropUrl + ')';
    crop.style.backgroundSize = (100 / win.sw) + '% ' + (100 / win.sh) + '%';
    crop.style.backgroundPosition = (win.sw >= 1 ? 0 : win.left / (1 - win.sw) * 100) + '% ' + (win.sh >= 1 ? 0 : win.top / (1 - win.sh) * 100) + '%';
  } else {
    crop.style.backgroundImage = cropUrl ? 'url(' + cropUrl + ')' : 'none';
    crop.style.backgroundSize = 'cover';
    crop.style.backgroundPosition = 'center';
  }
}
document.getElementById('rvFrame').onload = () => {
  const img = document.getElementById('rvFrame');
  document.getElementById('rvFrameSize').textContent = img.naturalWidth + ' × ' + img.naturalHeight;
  if (activeRun && reviewIndex >= 0) paintEvidence();
};

function openReview(idx) {
  if (!activeRun || !activeRun.ordered.length) return;
  reviewIndex = (idx + activeRun.ordered.length) % activeRun.ordered.length;
  const item = activeRun.ordered[reviewIndex];
  document.getElementById('reviewPane').classList.add('open');
  document.getElementById('rvTitle').textContent = 'Candidate #' + item.object_id;
  const tag = document.getElementById('rvStatus');
  tag.textContent = item.review_status;
  tag.className = 'status-tag s-' + item.review_status;
  document.getElementById('rvFrameCaption').textContent = 'frame @ ' + Number(item.best_source_seconds).toFixed(2) + ' s · ' + item.prompt;
  document.getElementById('rvGps').textContent = item.latitude === null ? 'unavailable' : Number(item.latitude).toFixed(5) + ', ' + Number(item.longitude).toFixed(5);
  document.getElementById('rvWhen').textContent = item.recorded_utc ? new Date(item.recorded_utc).toISOString().replace('T', ' · ').slice(0, 21) + ' UTC' : '';
  renderTriage(item);
  paintEvidence();
  renderQueue();
  const m = candidateMarkers.get(item.object_id);
  if (m) map.panTo(m.getLatLng());
}

function closeReview() {
  document.getElementById('reviewPane').classList.remove('open');
  reviewIndex = -1;
  if (activeRun) renderQueue();
  setTimeout(() => map.invalidateSize(), 0);
}

document.querySelectorAll('.view-toggle button').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    evidenceView = btn.dataset.view;
    if (reviewIndex >= 0) paintEvidence();
  };
});

async function setDecision(item, status) {
  const res = await api('/api/runs/' + activeRun.run_id + '/review', {
    method: 'POST',
    body: JSON.stringify({object_id: item.object_id, review_status: status})
  });
  item.review_status = res.review_status;
  if (candidateMarkers.has(item.object_id)) {
    candidateMarkers.get(item.object_id).setStyle({fillColor: markerColour(status)});
    candidateMarkers.get(item.object_id).setTooltipContent('#' + esc(item.object_id) + ' · ' + esc(status));
  }
  updateCounts();
}

async function decideAndAdvance(status) {
  if (!activeRun || reviewIndex < 0) return;
  await setDecision(activeRun.ordered[reviewIndex], status);
  openReview(reviewIndex + 1);
}
document.getElementById('rvConfirm').onclick = () => decideAndAdvance('confirmed');
document.getElementById('rvReject').onclick = () => decideAndAdvance('rejected');
document.getElementById('rvNext').onclick = () => openReview(reviewIndex + 1);
document.getElementById('rvPrev').onclick = () => openReview(reviewIndex - 1);
document.getElementById('rvClose').onclick = closeReview;
document.addEventListener('keydown', (e) => {
  if (reviewIndex < 0 || e.metaKey || e.ctrlKey || e.altKey) return;
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  const key = e.key.toLowerCase();
  if (key === 'c') { e.preventDefault(); decideAndAdvance('confirmed'); }
  else if (key === 'x') { e.preventDefault(); decideAndAdvance('rejected'); }
  else if (key === 'arrowright' || key === 'j') { e.preventDefault(); openReview(reviewIndex + 1); }
  else if (key === 'arrowleft' || key === 'k') { e.preventDefault(); openReview(reviewIndex - 1); }
  else if (key === 'escape') { e.preventDefault(); closeReview(); }
});

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
  document.getElementById('reportSection').style.display = 'flex';
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
            result_path = run_dir / f"result-{fidx:06d}.json"
            if result_path.is_file():
                detections = json.loads(result_path.read_text(encoding="utf-8")).get(
                    "detections", []
                )
                copy_item["box_xywh_normalized"] = _detection_box_for_observation(
                    detections, copy_item.get("object_id")
                )
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
                                max_candidates=(
                                    None
                                    if body.get("max_candidates") is None
                                    else int(body["max_candidates"])
                                ),
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
