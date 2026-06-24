from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import CSV_DATASET_PATH, DATASET_PATH, DEPLOY_DATASET_PATH, FRONTIER_DISTANCE_KM, PARTIDOS_AMBA_PATH


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not Path(DATASET_PATH).exists():
        candidates = [str(DEPLOY_DATASET_PATH), str(CSV_DATASET_PATH)]
        raise FileNotFoundError(
            "No se encontro el dataset de la app. "
            f"Rutas probadas: {candidates}. "
            "En deploy, subir data/processed/estaciones_streamlit.csv.gz."
        )
    df = pd.read_csv(DATASET_PATH, parse_dates=["periodo", "fecha_baja"], low_memory=False)
    if "activa_en_periodo" in df.columns:
        df["activa_en_periodo"] = df["activa_en_periodo"].astype(bool)
    if "tiene_desglose_impositivo" in df.columns:
        df["tiene_desglose_impositivo"] = df["tiene_desglose_impositivo"].astype(bool)
    return df


def get_partidos_geojson_version() -> int:
    path = Path(PARTIDOS_AMBA_PATH)
    if not path.exists():
        return 0
    return path.stat().st_mtime_ns


@st.cache_data(show_spinner=False)
def _read_geojson(path_str: str, version: int) -> dict | None:
    if version == 0:
        return None
    geojson = json.loads(Path(path_str).read_text(encoding="utf-8"))
    features = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        partido = properties.get("partido") or properties.get("shapeName") or ""
        if str(partido).strip().lower().startswith("comuna "):
            continue
        if partido:
            properties["partido"] = partido
        feature["properties"] = properties
        features.append(feature)
    geojson["features"] = features
    return geojson


def load_partidos_amba_geojson() -> dict | None:
    path = Path(PARTIDOS_AMBA_PATH)
    version = get_partidos_geojson_version()
    if version == 0:
        return None
    return _read_geojson(str(path), version)


@st.cache_data(show_spinner=False)
def get_filter_options() -> dict[str, list]:
    df = load_data()
    banderas = (
        df.groupby("bandera", dropna=False)["volumen"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return {
        "banderas": banderas,
        "productos": sorted(df["producto"].dropna().unique().tolist()),
        "periodos": sorted(pd.to_datetime(df["periodo"].dropna().unique()).tolist()),
        "tipos_negocio": sorted(df["tipo_negocio"].dropna().unique().tolist()),
        "provincias": sorted(df["provincia"].dropna().unique().tolist()),
    }


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    ring_length = len(ring)
    for i in range(ring_length):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % ring_length]
        intersects = ((y1 > lat) != (y2 > lat)) and (
            lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


def point_in_geometry(lon: float, lat: float, geometry: dict) -> bool:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        if not coordinates:
            return False
        if not point_in_ring(lon, lat, coordinates[0]):
            return False
        for hole in coordinates[1:]:
            if point_in_ring(lon, lat, hole):
                return False
        return True

    if geometry_type == "MultiPolygon":
        for polygon in coordinates:
            if not polygon:
                continue
            if not point_in_ring(lon, lat, polygon[0]):
                continue
            inside_hole = any(point_in_ring(lon, lat, hole) for hole in polygon[1:])
            if not inside_hole:
                return True
        return False

    return False


@st.cache_data(show_spinner=False)
def assign_partido_amba(rows: pd.DataFrame, geojson_version: int) -> pd.DataFrame:
    geojson = load_partidos_amba_geojson()
    enriched = rows.copy()
    enriched["partido_amba"] = pd.NA
    if geojson is None or enriched.empty:
        return enriched

    features = geojson.get("features", [])
    for index, row in enriched.iterrows():
        if row.get("provincia") != "Buenos Aires":
            continue
        if pd.isna(row.get("latitud")) or pd.isna(row.get("longitud")):
            continue

        lat = float(row["latitud"])
        lon = float(row["longitud"])
        for feature in features:
            bbox = feature.get("properties", {}).get("bbox")
            if not bbox:
                continue
            min_lon, min_lat, max_lon, max_lat = bbox
            if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
                continue
            if point_in_geometry(lon, lat, feature.get("geometry", {})):
                enriched.at[index, "partido_amba"] = feature.get("properties", {}).get("partido")
                break
    return enriched


@st.cache_data(show_spinner=False)
def get_localidades(provincias: tuple[str, ...]) -> list[str]:
    if not provincias:
        return []
    df = load_data()
    rows = df.loc[df["provincia"].isin(provincias), "localidad"].dropna().unique().tolist()
    return sorted(rows)


@st.cache_data(show_spinner=False)
def query_estaciones(
    banderas: tuple[str, ...],
    producto: str,
    periodo: pd.Timestamp,
    provincias: tuple[str, ...],
    localidades: tuple[str, ...],
    solo_activas: bool,
    tipos_negocio: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    df = load_data()
    filtered = df[
        (df["producto"] == producto)
        & (pd.to_datetime(df["periodo"]) == pd.Timestamp(periodo))
        & (df["bandera"].isin(banderas))
        & (df["provincia"].isin(provincias))
    ].copy()
    if localidades:
        filtered = filtered.loc[filtered["localidad"].isin(localidades)].copy()
    if solo_activas:
        filtered = filtered.loc[filtered["activa_en_periodo"]].copy()

    grouped = (
        filtered.groupby(
            [
                "nro_inscripcion",
                "operador",
                "bandera",
                "direccion",
                "localidad",
                "provincia",
                "producto",
                "latitud",
                "longitud",
            ],
            dropna=False,
        )
        .agg(
            volumen=("volumen", "sum"),
            precio_surtidor=("precio_surtidor", "mean"),
            precio_sin_impuestos=("precio_sin_impuestos", "mean"),
            precio_con_impuestos=("precio_con_impuestos", "mean"),
            variacion_impositiva=("variacion_impositiva", "mean"),
            tasa_vial=("tasa_vial", "mean"),
            tasa_municipal=("tasa_municipal", "mean"),
            ingresos_brutos=("ingresos_brutos", "mean"),
            impuesto_combustible_liquidos=("impuesto_combustible_liquidos", "mean"),
            impuesto_dioxido_carbono=("impuesto_dioxido_carbono", "mean"),
            iva=("iva", "mean"),
            fondo_fiduciario_gnc=("fondo_fiduciario_gnc", "mean"),
            carga_subnacional_informada_pct=("carga_subnacional_informada_pct", "mean"),
            tiene_desglose_impositivo=("tiene_desglose_impositivo", "max"),
            iibb_referencia_texto=("iibb_referencia_texto", "first"),
            iibb_referencia_min_pct=("iibb_referencia_min_pct", "mean"),
            iibb_referencia_max_pct=("iibb_referencia_max_pct", "mean"),
            tasa_vial_referencia_tipo=("tasa_vial_referencia_tipo", "first"),
            tasa_vial_referencia_valor=("tasa_vial_referencia_valor", "mean"),
        )
        .reset_index()
    )
    grouped["carga_subnacional_observable_pct"] = grouped[
        ["tasa_vial", "tasa_municipal", "ingresos_brutos"]
    ].sum(axis=1, min_count=1)
    grouped["carga_subnacional_comparable"] = grouped["carga_subnacional_observable_pct"].combine_first(
        grouped["tasa_vial_referencia_valor"]
    )
    grouped = assign_partido_amba(grouped, geojson_version=get_partidos_geojson_version())
    grouped["origen_frontera"] = grouped.apply(
        lambda row: "CABA"
        if row.get("provincia") == "Ciudad Autónoma de Buenos Aires"
        else f"Partido: {row.get('partido_amba')}" if pd.notna(row.get("partido_amba"))
        else f"Localidad: {row.get('localidad')}",
        axis=1,
    )
    return grouped


def calculate_radius(volume: pd.Series, radio_min_m: float = 80, radio_max_m: float = 1200) -> pd.Series:
    if volume.empty:
        return pd.Series(dtype="float64")
    vmin = float(volume.quantile(0.05))
    vmax = float(volume.quantile(0.95))
    if math.isclose(vmax, vmin):
        return pd.Series(np.full(len(volume), radio_min_m), index=volume.index)
    clipped = volume.clip(lower=vmin, upper=vmax)
    normalized = (np.sqrt(clipped) - math.sqrt(vmin)) / (math.sqrt(vmax) - math.sqrt(vmin))
    emphasized = np.power(normalized, 1.35)
    return radio_min_m + emphasized * (radio_max_m - radio_min_m)


def haversine_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    earth_radius_km = 6371.0
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    lat1 = lat_rad[:, None]
    lat2 = lat_rad[None, :]
    lon1 = lon_rad[:, None]
    lon2 = lon_rad[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def haversine_pair(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


@st.cache_data(show_spinner=False)
def compute_frontier_pairs(df: pd.DataFrame, max_distance_km: float = FRONTIER_DISTANCE_KM) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    working = df.dropna(subset=["latitud", "longitud", "carga_subnacional_comparable"]).reset_index(drop=True).copy()
    if len(working) < 2:
        return pd.DataFrame()

    lat_step = max_distance_km / 111.0
    mean_lat_rad = math.radians(float(working["latitud"].mean()))
    lon_step = max_distance_km / max(111.0 * math.cos(mean_lat_rad), 0.1)

    working["lat_bucket"] = np.floor(working["latitud"] / lat_step).astype(int)
    working["lon_bucket"] = np.floor(working["longitud"] / lon_step).astype(int)

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in working[["lat_bucket", "lon_bucket"]].iterrows():
        buckets[(int(row["lat_bucket"]), int(row["lon_bucket"]))].append(index)

    rows: list[dict[str, object]] = []
    for i, left in working.iterrows():
        lat_bucket = int(left["lat_bucket"])
        lon_bucket = int(left["lon_bucket"])
        candidate_indices: list[int] = []
        for lat_offset in (-1, 0, 1):
            for lon_offset in (-1, 0, 1):
                candidate_indices.extend(buckets.get((lat_bucket + lat_offset, lon_bucket + lon_offset), []))

        for j in candidate_indices:
            if j <= i:
                continue

            right = working.iloc[j]
            distance_km = haversine_pair(
                float(left["latitud"]),
                float(left["longitud"]),
                float(right["latitud"]),
                float(right["longitud"]),
            )
            if distance_km > max_distance_km:
                continue

            provinces = {left.get("provincia"), right.get("provincia")}
            if left.get("origen_frontera") == right.get("origen_frontera"):
                continue
            if provinces == {"Buenos Aires", "Ciudad Autónoma de Buenos Aires"}:
                tipo_frontera = "CABA vs Provincia"
            elif (
                left.get("provincia") == "Buenos Aires"
                and right.get("provincia") == "Buenos Aires"
                and pd.notna(left.get("partido_amba"))
                and pd.notna(right.get("partido_amba"))
                and left.get("partido_amba") != right.get("partido_amba")
            ):
                tipo_frontera = "Interpartido BA"
            else:
                continue

            carga_left = left.get("carga_subnacional_comparable")
            carga_right = right.get("carga_subnacional_comparable")
            if pd.isna(carga_left) or pd.isna(carga_right):
                continue
            if math.isclose(float(carga_left), float(carga_right), abs_tol=0.01):
                continue

            if float(carga_left) <= float(carga_right):
                low, high = left, right
            else:
                low, high = right, left

            rows.append(
                {
                    "distancia_km": round(distance_km, 3),
                    "tipo_frontera": tipo_frontera,
                    "origen_menor_carga": low.get("origen_frontera"),
                    "origen_mayor_carga": high.get("origen_frontera"),
                    "estacion_menor_carga": f"{low['operador']} ({low['localidad']})",
                    "estacion_mayor_carga": f"{high['operador']} ({high['localidad']})",
                    "provincia_menor_carga": low["provincia"],
                    "provincia_mayor_carga": high["provincia"],
                    "partido_menor_carga": low.get("partido_amba"),
                    "partido_mayor_carga": high.get("partido_amba"),
                    "bandera_menor_carga": low["bandera"],
                    "bandera_mayor_carga": high["bandera"],
                    "volumen_menor_carga": float(low["volumen"]),
                    "volumen_mayor_carga": float(high["volumen"]),
                    "precio_menor_carga": float(low["precio_surtidor"]),
                    "precio_mayor_carga": float(high["precio_surtidor"]),
                    "carga_menor_pct": float(low["carga_subnacional_comparable"]),
                    "carga_mayor_pct": float(high["carga_subnacional_comparable"]),
                    "gap_carga_pct": float(high["carga_subnacional_comparable"] - low["carga_subnacional_comparable"]),
                    "gap_precio": float(high["precio_surtidor"] - low["precio_surtidor"]),
                    "gap_volumen": float(low["volumen"] - high["volumen"]),
                    "favorece_hipotesis": bool(float(low["volumen"]) > float(high["volumen"])),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["distancia_km", "gap_carga_pct"], ascending=[True, False]).reset_index(drop=True)
