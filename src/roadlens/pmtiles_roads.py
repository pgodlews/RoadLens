"""Offline street lookup and reverse geocoding from local PMTiles v3 MVT archives."""

from __future__ import annotations

import gzip
import math
import struct
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class RoadFeature:
    name: str
    kind: str | None
    ref: str | None
    lines: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class ReverseGeocodeMatch:
    street_name: str | None
    road_ref: str | None
    place_name: str | None
    distance_m: float


class StreetGeocoder(Protocol):
    def find_roads(
        self,
        street_name: str,
        tiles_or_points: Sequence[tuple[float, float]],
    ) -> list[RoadFeature]: ...

    def reverse_road(
        self,
        latitude: float,
        longitude: float,
        max_distance_m: float = 80.0,
    ) -> ReverseGeocodeMatch: ...


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def point_to_segment_distance_m(
    lat: float,
    lon: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    cos_lat = math.cos(math.radians(lat))
    px, py = 0.0, 0.0
    ax = math.radians(lon1 - lon) * EARTH_RADIUS_M * cos_lat
    ay = math.radians(lat1 - lat) * EARTH_RADIUS_M
    bx = math.radians(lon2 - lon) * EARTH_RADIUS_M * cos_lat
    by = math.radians(lat2 - lat) * EARTH_RADIUS_M
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_to_polyline_distance_m(
    lat: float,
    lon: float,
    polyline: Sequence[tuple[float, float]],
) -> float:
    if not polyline:
        return math.inf
    if len(polyline) == 1:
        return haversine_m(lat, lon, polyline[0][0], polyline[0][1])
    return min(
        point_to_segment_distance_m(
            lat,
            lon,
            polyline[i][0],
            polyline[i][1],
            polyline[i + 1][0],
            polyline[i + 1][1],
        )
        for i in range(len(polyline) - 1)
    )


def latlon_to_zxy(lat: float, lon: float, zoom: int) -> tuple[int, int, int]:
    n = 1 << zoom
    clamped_lat = max(-85.05112878, min(85.05112878, lat))
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(clamped_lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return zoom, max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_xy_to_latlon(
    zoom: int, x: int, y: int, px: float, py: float, extent: int = 4096
) -> tuple[float, float]:
    n = 1 << zoom
    lon = (x + px / extent) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + py / extent) / n)))
    return math.degrees(lat_rad), lon


def zxy_to_tile_id(zoom: int, x: int, y: int) -> int:
    if zoom == 0:
        return 0
    acc = ((1 << (2 * zoom)) - 1) // 3
    tx, ty = x, y
    d = 0
    s = 1 << (zoom - 1)
    while s > 0:
        rx = 1 if (tx & s) > 0 else 0
        ry = 1 if (ty & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                tx = s - 1 - tx
                ty = s - 1 - ty
            tx, ty = ty, tx
        s >>= 1
    return acc + d


def _normalize_street_name(name: str) -> str:
    cleaned = " ".join(name.strip().lower().replace(".", "").split())
    abbreviations = {
        " rd": " road",
        " st": " street",
        " ln": " lane",
        " ave": " avenue",
        " dr": " drive",
    }
    for short, full in abbreviations.items():
        if cleaned.endswith(short):
            cleaned = cleaned[: -len(short)] + full
    return cleaned


class StaticStreetIndex:
    """In-memory street index for deterministic offline tests and custom GeoJSON."""

    def __init__(
        self,
        roads: Sequence[RoadFeature],
        places: Sequence[tuple[str, float, float]] = (),
    ):
        self._roads = tuple(roads)
        self._places = tuple(places)

    def find_roads(
        self,
        street_name: str,
        tiles_or_points: Sequence[tuple[float, float]],
    ) -> list[RoadFeature]:
        del tiles_or_points
        needle = _normalize_street_name(street_name)
        if not needle:
            return []
        matches = []
        for road in self._roads:
            candidate = _normalize_street_name(road.name)
            ref = (road.ref or "").strip().lower()
            if needle == candidate or needle in candidate or (ref and needle == ref):
                matches.append(road)
        return matches

    def reverse_road(
        self,
        latitude: float,
        longitude: float,
        max_distance_m: float = 80.0,
    ) -> ReverseGeocodeMatch:
        best_road: RoadFeature | None = None
        best_dist = math.inf
        for road in self._roads:
            if not road.name and not road.ref:
                continue
            for line in road.lines:
                dist = point_to_polyline_distance_m(latitude, longitude, line)
                if dist < best_dist:
                    best_dist = dist
                    best_road = road
        nearest_place = None
        if self._places:
            nearest_place = min(
                self._places,
                key=lambda item: haversine_m(latitude, longitude, item[1], item[2]),
            )[0]
        if best_road is None or best_dist > max_distance_m:
            return ReverseGeocodeMatch(None, None, nearest_place, best_dist)
        return ReverseGeocodeMatch(
            street_name=best_road.name or None,
            road_ref=best_road.ref,
            place_name=nearest_place,
            distance_m=best_dist,
        )


class PmtilesStreetIndex:
    """Read Protomaps/OpenMapTiles MVT roads & places from local PMTiles v3."""

    def __init__(self, pmtiles_path: Path):
        if not pmtiles_path.is_file():
            raise FileNotFoundError(pmtiles_path)
        self.path = pmtiles_path
        with self.path.open("rb") as handle:
            header = handle.read(127)
        if len(header) < 127 or header[:7] != b"PMTiles" or header[7] != 3:
            raise ValueError(f"Unsupported PMTiles v3 header in {pmtiles_path}")
        (
            self._root_offset,
            self._root_length,
            _meta_offset,
            _meta_length,
            self._leaf_offset,
            _leaf_length,
            self._tile_offset,
            _tile_length,
        ) = struct.unpack("<8Q", header[8:72])
        self._internal_compression = header[97]
        self._tile_compression = header[98]
        self.min_zoom = header[100]
        self.max_zoom = header[101]
        self._tile_cache: dict[
            tuple[int, int, int],
            tuple[list[RoadFeature], list[tuple[str, float, float]]],
        ] = {}

    def _decompress(self, raw: bytes, compression: int) -> bytes:
        if compression == 1:
            return raw
        if compression == 2:
            return gzip.decompress(raw)
        if compression == 3:
            return zlib.decompress(raw)
        raise ValueError(f"Unsupported PMTiles compression code {compression}")

    def read_tile_bytes(self, zoom: int, x: int, y: int) -> bytes | None:
        tile_id = zxy_to_tile_id(zoom, x, y)
        dir_offset, dir_length = self._root_offset, self._root_length
        with self.path.open("rb") as handle:
            for _ in range(5):
                handle.seek(dir_offset)
                entries = _decode_pmtiles_directory(
                    self._decompress(
                        handle.read(dir_length), self._internal_compression
                    )
                )
                idx = _find_directory_entry(entries, tile_id)
                if idx < 0:
                    return None
                entry_id, run_length, offset, length = entries[idx]
                if run_length == 0:
                    dir_offset = self._leaf_offset + offset
                    dir_length = length
                elif tile_id < entry_id + run_length:
                    handle.seek(self._tile_offset + offset)
                    return self._decompress(handle.read(length), self._tile_compression)
                else:
                    return None
        return None

    def _load_tile(
        self, zoom: int, x: int, y: int
    ) -> tuple[list[RoadFeature], list[tuple[str, float, float]]]:
        key = (zoom, x, y)
        if key in self._tile_cache:
            return self._tile_cache[key]
        raw = self.read_tile_bytes(zoom, x, y)
        if raw is None:
            parsed: tuple[list[RoadFeature], list[tuple[str, float, float]]] = ([], [])
        else:
            parsed = _decode_mvt_roads_and_places(raw, zoom, x, y)
        if len(self._tile_cache) >= 256:
            self._tile_cache.clear()
        self._tile_cache[key] = parsed
        return parsed

    def find_roads(
        self,
        street_name: str,
        tiles_or_points: Sequence[tuple[float, float]],
    ) -> list[RoadFeature]:
        needle = _normalize_street_name(street_name)
        if not needle or not tiles_or_points:
            return []
        zoom = min(15, self.max_zoom)
        tile_coords: set[tuple[int, int]] = set()
        for lat, lon in tiles_or_points:
            _, tx, ty = latlon_to_zxy(lat, lon, zoom)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = tx + dx, ty + dy
                    if 0 <= nx < (1 << zoom) and 0 <= ny < (1 << zoom):
                        tile_coords.add((nx, ny))
        matched: list[RoadFeature] = []
        for tx, ty in sorted(tile_coords):
            roads, _ = self._load_tile(zoom, tx, ty)
            for road in roads:
                candidate = _normalize_street_name(road.name)
                ref = (road.ref or "").strip().lower()
                if (
                    needle == candidate
                    or needle in candidate
                    or (ref and needle == ref)
                ):
                    matched.append(road)
        return matched

    def reverse_road(
        self,
        latitude: float,
        longitude: float,
        max_distance_m: float = 80.0,
    ) -> ReverseGeocodeMatch:
        zoom = min(15, self.max_zoom)
        _, cx, cy = latlon_to_zxy(latitude, longitude, zoom)
        best_road: RoadFeature | None = None
        best_dist = math.inf
        places: list[tuple[str, float, float]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < (1 << zoom) and 0 <= ny < (1 << zoom)):
                    continue
                tile_roads, tile_places = self._load_tile(zoom, nx, ny)
                places.extend(tile_places)
                for road in tile_roads:
                    if not road.name and not road.ref:
                        continue
                    for line in road.lines:
                        dist = point_to_polyline_distance_m(latitude, longitude, line)
                        if dist < best_dist:
                            best_dist = dist
                            best_road = road
        nearest_place = None
        if places:
            nearest_place = min(
                places,
                key=lambda item: haversine_m(latitude, longitude, item[1], item[2]),
            )[0]
        if best_road is None or best_dist > max_distance_m:
            return ReverseGeocodeMatch(None, None, nearest_place, best_dist)
        return ReverseGeocodeMatch(
            street_name=best_road.name or None,
            road_ref=best_road.ref,
            place_name=nearest_place,
            distance_m=best_dist,
        )


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7


def _decode_pmtiles_directory(
    raw: bytes,
) -> list[tuple[int, int, int, int]]:
    pos = 0
    count, pos = _read_varint(raw, pos)
    tile_ids: list[int] = []
    last_id = 0
    for _ in range(count):
        delta, pos = _read_varint(raw, pos)
        last_id += delta
        tile_ids.append(last_id)
    run_lengths: list[int] = []
    for _ in range(count):
        value, pos = _read_varint(raw, pos)
        run_lengths.append(value)
    lengths: list[int] = []
    for _ in range(count):
        value, pos = _read_varint(raw, pos)
        lengths.append(value)
    offsets: list[int] = []
    for idx in range(count):
        value, pos = _read_varint(raw, pos)
        if value == 0 and idx > 0:
            offsets.append(offsets[idx - 1] + lengths[idx - 1])
        else:
            offsets.append(value - 1)
    return list(zip(tile_ids, run_lengths, offsets, lengths, strict=True))


def _find_directory_entry(
    entries: Sequence[tuple[int, int, int, int]], tile_id: int
) -> int:
    lo, hi = 0, len(entries) - 1
    found = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if entries[mid][0] <= tile_id:
            found = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return found


def _iter_protobuf_fields(buf: bytes) -> Iterator[tuple[int, int, Any]]:
    pos = 0
    limit = len(buf)
    while pos < limit:
        tag, pos = _read_varint(buf, pos)
        field_num, wire_type = tag >> 3, tag & 0x07
        if wire_type == 0:
            val, pos = _read_varint(buf, pos)
        elif wire_type == 1:
            val = buf[pos : pos + 8]
            pos += 8
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            val = buf[pos : pos + length]
            pos += length
        elif wire_type == 5:
            val = buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported MVT protobuf wire type {wire_type}")
        yield field_num, wire_type, val


def _unpack_packed_uints(raw: bytes) -> list[int]:
    pos = 0
    values: list[int] = []
    limit = len(raw)
    while pos < limit:
        value, pos = _read_varint(raw, pos)
        values.append(value)
    return values


def _decode_mvt_value(raw: bytes) -> Any:
    for field_num, _wire, val in _iter_protobuf_fields(raw):
        if field_num == 1:
            return val.decode("utf-8", errors="replace")
        if field_num == 2:
            return struct.unpack("<f", val)[0]
        if field_num == 3:
            return struct.unpack("<d", val)[0]
        if field_num in (4, 5):
            return val
        if field_num == 6:
            return (val >> 1) ^ -(val & 1)
        if field_num == 7:
            return bool(val)
    return None


def _decode_mvt_points(
    geom: Sequence[int], zoom: int, x: int, y: int, extent: int
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    cx = cy = 0
    idx = 0
    while idx < len(geom):
        cmd_int = geom[idx]
        idx += 1
        cmd, count = cmd_int & 0x07, cmd_int >> 3
        if cmd in (1, 2):
            for _ in range(count):
                dx = (geom[idx] >> 1) ^ -(geom[idx] & 1)
                dy = (geom[idx + 1] >> 1) ^ -(geom[idx + 1] & 1)
                idx += 2
                cx += dx
                cy += dy
                points.append(tile_xy_to_latlon(zoom, x, y, cx, cy, extent))
    return points


def _decode_mvt_linestrings(
    geom: Sequence[int], zoom: int, x: int, y: int, extent: int
) -> tuple[tuple[tuple[float, float], ...], ...]:
    lines: list[tuple[tuple[float, float], ...]] = []
    current: list[tuple[float, float]] = []
    cx = cy = 0
    idx = 0
    while idx < len(geom):
        cmd_int = geom[idx]
        idx += 1
        cmd, count = cmd_int & 0x07, cmd_int >> 3
        if cmd == 1:
            for _ in range(count):
                if len(current) >= 2:
                    lines.append(tuple(current))
                dx = (geom[idx] >> 1) ^ -(geom[idx] & 1)
                dy = (geom[idx + 1] >> 1) ^ -(geom[idx + 1] & 1)
                idx += 2
                cx += dx
                cy += dy
                current = [tile_xy_to_latlon(zoom, x, y, cx, cy, extent)]
        elif cmd == 2:
            for _ in range(count):
                dx = (geom[idx] >> 1) ^ -(geom[idx] & 1)
                dy = (geom[idx + 1] >> 1) ^ -(geom[idx + 1] & 1)
                idx += 2
                cx += dx
                cy += dy
                current.append(tile_xy_to_latlon(zoom, x, y, cx, cy, extent))
        elif cmd == 7 and current:
            current.append(current[0])
    if len(current) >= 2:
        lines.append(tuple(current))
    return tuple(lines)


def _decode_mvt_roads_and_places(
    raw: bytes, zoom: int, x: int, y: int
) -> tuple[list[RoadFeature], list[tuple[str, float, float]]]:
    roads: list[RoadFeature] = []
    places: list[tuple[str, float, float]] = []
    for field_num, _wire, layer_raw in _iter_protobuf_fields(raw):
        if field_num != 3:
            continue
        layer_name = ""
        keys: list[str] = []
        values: list[Any] = []
        features: list[bytes] = []
        extent = 4096
        for l_num, _l_wire, l_val in _iter_protobuf_fields(layer_raw):
            if l_num == 1:
                layer_name = l_val.decode("utf-8", errors="replace")
            elif l_num == 2:
                features.append(l_val)
            elif l_num == 3:
                keys.append(l_val.decode("utf-8", errors="replace"))
            elif l_num == 4:
                values.append(_decode_mvt_value(l_val))
            elif l_num == 5:
                extent = int(l_val)
        if layer_name not in (
            "roads",
            "transportation",
            "transportation_name",
            "places",
        ):
            continue
        for feat_raw in features:
            tags: list[int] = []
            geom_type = 0
            geom: list[int] = []
            for f_num, f_wire, f_val in _iter_protobuf_fields(feat_raw):
                if f_num == 2:
                    if f_wire == 2:
                        tags.extend(_unpack_packed_uints(f_val))
                    else:
                        tags.append(int(f_val))
                elif f_num == 3:
                    geom_type = int(f_val)
                elif f_num == 4:
                    if f_wire == 2:
                        geom.extend(_unpack_packed_uints(f_val))
                    else:
                        geom.append(int(f_val))
            props = {
                keys[tags[i]]: values[tags[i + 1]]
                for i in range(0, len(tags) - 1, 2)
                if tags[i] < len(keys) and tags[i + 1] < len(values)
            }
            name = str(props.get("name") or props.get("name:en") or "").strip()
            ref = str(props.get("ref") or "").strip() or None
            kind = str(props.get("kind") or props.get("class") or "").strip() or None
            if layer_name == "places":
                if name and geom_type == 1:
                    pts = _decode_mvt_points(geom, zoom, x, y, extent)
                    if pts:
                        places.append((name, pts[0][0], pts[0][1]))
            elif geom_type == 2 and (name or ref):
                lines = _decode_mvt_linestrings(geom, zoom, x, y, extent)
                if lines:
                    roads.append(
                        RoadFeature(name=name, kind=kind, ref=ref, lines=lines)
                    )
    return roads, places


def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("Varint must be nonnegative")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def build_minimal_pmtiles_v3(
    tiles: Iterable[tuple[int, int, int, bytes]],
    max_zoom: int = 15,
) -> bytes:
    """Create a tiny spec-compliant PMTiles v3 archive for offline unit tests."""
    ordered = sorted(
        (zxy_to_tile_id(z, x, y), gzip.compress(payload)) for z, x, y, payload in tiles
    )
    dir_buf = bytearray(encode_varint(len(ordered)))
    prev_id = 0
    for tile_id, _ in ordered:
        dir_buf.extend(encode_varint(tile_id - prev_id))
        prev_id = tile_id
    for _ in ordered:
        dir_buf.extend(encode_varint(1))
    for _, compressed in ordered:
        dir_buf.extend(encode_varint(len(compressed)))
    for idx, _ in enumerate(ordered):
        dir_buf.extend(encode_varint(1 if idx == 0 else 0))
    root_bytes = gzip.compress(bytes(dir_buf))
    meta_bytes = gzip.compress(b"{}")
    tile_data = b"".join(compressed for _, compressed in ordered)
    root_offset = 127
    root_length = len(root_bytes)
    meta_offset = root_offset + root_length
    meta_length = len(meta_bytes)
    tile_offset = meta_offset + meta_length
    tile_length = len(tile_data)
    header = bytearray(127)
    header[:7] = b"PMTiles"
    header[7] = 3
    struct.pack_into(
        "<8Q",
        header,
        8,
        root_offset,
        root_length,
        meta_offset,
        meta_length,
        tile_offset,
        0,
        tile_offset,
        tile_length,
    )
    struct.pack_into("<3Q", header, 72, len(ordered), len(ordered), len(ordered))
    header[96] = 0
    header[97] = 2  # gzip internal
    header[98] = 2  # gzip tile
    header[99] = 1  # MVT
    header[100] = 0
    header[101] = max_zoom
    return bytes(header) + root_bytes + meta_bytes + tile_data
