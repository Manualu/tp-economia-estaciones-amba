from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from config import (
    ARGENTINA_BBOX,
    BANDERA_ALIASES,
    CLEANING_REPORT_PATH,
    CSV_DATASET_PATH,
    IIBB_PATH,
    OVERRIDES_PATH,
    PRODUCT_ALIASES,
    PRODUCT_TO_TASA_VIAL_COLUMN,
    PROVINCE_ALIASES,
    RAW_DATA_PATH,
    TASA_VIAL_PATH,
)


RENAME_MAP = {
    "Período": "periodo",
    "Operador": "operador",
    "Nro Inscripción": "nro_inscripcion",
    "Bandera": "bandera",
    "Fecha de baja": "fecha_baja",
    "CUIT": "cuit",
    "Tipo Negocio": "tipo_negocio",
    "Dirección": "direccion",
    "Localidad": "localidad",
    "Provincia": "provincia",
    "Producto": "producto",
    "Canal de Comercialización": "canal_comercializacion",
    "Precio sin impuestos": "precio_sin_impuestos",
    "Precio con impuestos": "precio_con_impuestos",
    "Variación impositiva": "variacion_impositiva",
    "Volumen": "volumen",
    "Precio surtidor": "precio_surtidor",
    "NO Movimientos": "no_movimientos",
    "Excentos": "excentos",
    "Impuesto Combustible Líquidos": "impuesto_combustible_liquidos",
    "Impuesto Dióxido Carbono": "impuesto_dioxido_carbono",
    "Tasa Vial": "tasa_vial",
    "tasa Municipal": "tasa_municipal",
    "Ingresos Brutos": "ingresos_brutos",
    "Iva": "iva",
    "Fondo fiduciario GNC": "fondo_fiduciario_gnc",
    "Impuesto Combustible Líquido": "impuesto_combustible_liquido",
    "precios-en-surtidor-resolucin-3142016.latitud": "latitud_raw",
    "precios-en-surtidor-resolucin-3142016.longitud": "longitud_raw",
    "precios-en-surtidor-resolucin-3142016.geojson": "geojson",
}

READ_COLUMNS = [
    "Período",
    "Operador",
    "Nro Inscripción",
    "Bandera",
    "Fecha de baja",
    "CUIT",
    "Tipo Negocio",
    "Dirección",
    "Localidad",
    "Provincia",
    "Producto",
    "Canal de Comercialización",
    "Precio sin impuestos",
    "Precio con impuestos",
    "Variación impositiva",
    "Volumen",
    "Precio surtidor",
    "Impuesto Combustible Líquidos",
    "Impuesto Dióxido Carbono",
    "Tasa Vial",
    "tasa Municipal",
    "Ingresos Brutos",
    "Iva",
    "Fondo fiduciario GNC",
    "precios-en-surtidor-resolucin-3142016.latitud",
    "precios-en-surtidor-resolucin-3142016.longitud",
    "precios-en-surtidor-resolucin-3142016.geojson",
]

NUMERIC_COLUMNS = [
    "precio_sin_impuestos",
    "precio_con_impuestos",
    "variacion_impositiva",
    "volumen",
    "precio_surtidor",
    "impuesto_combustible_liquidos",
    "impuesto_dioxido_carbono",
    "tasa_vial",
    "tasa_municipal",
    "ingresos_brutos",
    "iva",
    "fondo_fiduciario_gnc",
    "impuesto_combustible_liquido",
]

STRING_CATEGORY_COLUMNS = [
    "bandera",
    "producto",
    "tipo_negocio",
    "canal_comercializacion",
    "provincia",
]


@dataclass
class CleaningStats:
    initial_rows: int
    hard_drops: int = 0
    logic_drops: int = 0
    iqr_drops: int = 0
    override_drops: int = 0
    override_updates: int = 0
    geographic_drops: int = 0
    final_rows: int = 0


def strip_accents(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def snake_case(value: Any) -> str:
    text = strip_accents(value)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def normalize_for_lookup(value: Any) -> str:
    return strip_accents(value).lower()


def parse_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.strip()
    both_separators = cleaned.str.contains(",", regex=False, na=False) & cleaned.str.contains(".", regex=False, na=False)
    cleaned = cleaned.where(~both_separators, cleaned.str.replace(".", "", regex=False))
    cleaned = cleaned.str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_period_value(value: Any) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    normalized = normalize_for_lookup(text).replace("_", " ").replace("/", "-")
    month_aliases = {
        "ene": "01",
        "feb": "02",
        "mar": "03",
        "abr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "ago": "08",
        "sep": "09",
        "set": "09",
        "oct": "10",
        "nov": "11",
        "dic": "12",
    }

    if re.fullmatch(r"\d{4}-\d{2}", normalized):
        return pd.to_datetime(f"{normalized}-01", errors="coerce")
    if re.fullmatch(r"\d{6}", normalized):
        return pd.to_datetime(f"{normalized[:4]}-{normalized[4:]}-01", errors="coerce")

    for alias, month in month_aliases.items():
        if normalized.startswith(f"{alias}-"):
            year = normalized.split("-")[-1]
            return pd.to_datetime(f"{year}-{month}-01", errors="coerce")

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed.year, parsed.month, 1)


def canonical_bandera(value: Any) -> str:
    if pd.isna(value):
        raw = ""
    else:
        raw = str(value).strip()
    lookup = normalize_for_lookup(raw)
    for canonical, aliases in BANDERA_ALIASES.items():
        alias_values = {normalize_for_lookup(alias) for alias in aliases if alias is not None}
        if lookup in alias_values:
            return canonical
    if not raw:
        return "Blanca"
    return raw.title()


def canonical_provincia(value: Any) -> str:
    if pd.isna(value):
        raw = ""
    else:
        raw = str(value).strip()
    if not raw:
        return raw
    upper = strip_accents(raw).upper()
    if upper in PROVINCE_ALIASES:
        return PROVINCE_ALIASES[upper]
    return raw.title()


def canonical_producto(value: Any) -> str:
    if pd.isna(value):
        raw = ""
    else:
        raw = str(value).strip()
    if not raw:
        return raw
    lookup = normalize_for_lookup(raw)
    return PRODUCT_ALIASES.get(lookup, raw.title())


def canonical_localidad(value: Any) -> str:
    if pd.isna(value):
        raw = ""
    else:
        raw = str(value).strip()
    return raw.title()


def parse_geojson_point(value: Any) -> tuple[float | None, float | None]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan, np.nan
    try:
        payload = json.loads(value)
        coordinates = payload.get("coordinates", [])
        if len(coordinates) >= 2:
            lon = float(coordinates[0])
            lat = float(coordinates[1])
            return lat, lon
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.nan, np.nan
    return np.nan, np.nan


def scale_coords_to_argentina(lat_raw: Any, lon_raw: Any) -> tuple[float | None, float | None]:
    if pd.isna(lat_raw) or pd.isna(lon_raw):
        return np.nan, np.nan
    try:
        lat_base = float(lat_raw)
        lon_base = float(lon_raw)
    except (TypeError, ValueError):
        return np.nan, np.nan

    for scale in range(0, 16):
        factor = 10**scale
        lat = lat_base / factor
        lon = lon_base / factor
        if (
            ARGENTINA_BBOX["lat_min"] <= lat <= ARGENTINA_BBOX["lat_max"]
            and ARGENTINA_BBOX["lon_min"] <= lon <= ARGENTINA_BBOX["lon_max"]
        ):
            return lat, lon
    return np.nan, np.nan


def build_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    extracted = df["geojson"].astype("string").str.extract(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    df["longitud"] = pd.to_numeric(extracted[0], errors="coerce")
    df["latitud"] = pd.to_numeric(extracted[1], errors="coerce")

    missing_geo = df["latitud"].isna() | df["longitud"].isna()
    scaled = df.loc[missing_geo, ["latitud_raw", "longitud_raw"]].apply(
        lambda row: scale_coords_to_argentina(row["latitud_raw"], row["longitud_raw"]),
        axis=1,
    )
    if not scaled.empty:
        df.loc[missing_geo, "latitud"] = pd.to_numeric(scaled.apply(lambda item: item[0]), errors="coerce").to_numpy()
        df.loc[missing_geo, "longitud"] = pd.to_numeric(scaled.apply(lambda item: item[1]), errors="coerce").to_numpy()
    return df


def build_cleaning_entry(
    row: pd.Series,
    motivo: str,
    regla: str,
    valor_problematico: str,
    accion: str,
) -> dict[str, Any]:
    periodo = row.get("periodo")
    if pd.notna(periodo):
        periodo = pd.Timestamp(periodo).strftime("%Y-%m")
    else:
        periodo = None
    return {
        "motivo": motivo,
        "regla": regla,
        "nro_inscripcion": row.get("nro_inscripcion"),
        "periodo": periodo,
        "producto": row.get("producto"),
        "bandera": row.get("bandera"),
        "valor_problematico": valor_problematico,
        "accion": accion,
    }


def apply_drop_rule(
    df: pd.DataFrame,
    mask: pd.Series,
    motivo: str,
    regla: str,
    value_column: str | None,
    report_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    matched = df.loc[mask].copy()
    for _, row in matched.iterrows():
        value = row.get(value_column) if value_column else None
        report_rows.append(
            build_cleaning_entry(
                row=row,
                motivo=motivo,
                regla=regla,
                valor_problematico=str(value),
                accion="dropped",
            )
        )
    return df.loc[~mask].copy()


def load_iibb_reference() -> pd.DataFrame:
    if not IIBB_PATH.exists():
        return pd.DataFrame()

    raw = pd.read_excel(IIBB_PATH, header=None)
    rows: list[dict[str, Any]] = []
    for item in raw.itertuples(index=False):
        values = [value for value in item if pd.notna(value)]
        if len(values) < 5:
            continue
        if values[0] in {"Jurisdicción", "Región Centro", "Región Patagónica", "Cuyo", "NOA", "NEA"}:
            continue
        if str(values[0]).startswith("Nota:"):
            continue
        provincia = canonical_provincia(values[0])
        text = str(values[1]).strip()
        rates = [float(match.replace(",", ".")) for match in re.findall(r"(\d+,\d+)", text)]
        rows.append(
            {
                "provincia": provincia,
                "iibb_referencia_texto": text,
                "iibb_referencia_min_pct": min(rates) if rates else None,
                "iibb_referencia_max_pct": max(rates) if rates else None,
                "iibb_referencia_fuente": values[3],
                "iibb_referencia_confianza": values[4],
            }
        )
    return pd.DataFrame(rows)


def load_tasa_vial_reference() -> pd.DataFrame:
    if not TASA_VIAL_PATH.exists():
        return pd.DataFrame()

    raw = pd.read_excel(TASA_VIAL_PATH, header=None)
    rows: list[dict[str, Any]] = []
    table_started = False
    current_type = None
    for item in raw.itertuples(index=False):
        values = [value for value in item if pd.notna(value)]
        if not values:
            continue
        first = str(values[0]).strip()
        if first == "Municipio":
            table_started = True
            continue
        if not table_started:
            continue
        if first == "Municipios con monto fijo por litro (en $ por litro)":
            current_type = "Monto fijo"
            continue
        if str(first).startswith("Datos relevados"):
            break
        if len(values) < 9:
            continue
        row_type = values[2] if values[2] else current_type
        rows.append(
            {
                "localidad": canonical_localidad(values[0]),
                "provincia": canonical_provincia(values[1]),
                "tasa_vial_referencia_tipo": row_type,
                "diesel_g2": pd.to_numeric(values[3], errors="coerce"),
                "diesel_g3": pd.to_numeric(values[4], errors="coerce"),
                "nafta_g2": pd.to_numeric(values[5], errors="coerce"),
                "nafta_g3": pd.to_numeric(values[6], errors="coerce"),
                "otros": pd.to_numeric(values[7], errors="coerce"),
                "gnc": pd.to_numeric(values[8], errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def attach_reference_tables(df: pd.DataFrame) -> pd.DataFrame:
    iibb = load_iibb_reference()
    if not iibb.empty:
        df = df.merge(iibb, on="provincia", how="left")

    tasa_vial = load_tasa_vial_reference()
    if not tasa_vial.empty:
        tasa_vial["join_key"] = tasa_vial["provincia"] + "|" + tasa_vial["localidad"]
        df["join_key"] = df["provincia"] + "|" + df["localidad"]
        df = df.merge(
            tasa_vial[
                [
                    "join_key",
                    "tasa_vial_referencia_tipo",
                    "diesel_g2",
                    "diesel_g3",
                    "nafta_g2",
                    "nafta_g3",
                    "otros",
                    "gnc",
                ]
            ],
            on="join_key",
            how="left",
        )
        df["tasa_vial_referencia_valor"] = df.apply(resolve_tasa_vial_reference, axis=1)
        df = df.drop(columns=["join_key"])
    else:
        df["tasa_vial_referencia_tipo"] = pd.NA
        df["tasa_vial_referencia_valor"] = pd.NA

    return df


def resolve_tasa_vial_reference(row: pd.Series) -> float | None:
    column = PRODUCT_TO_TASA_VIAL_COLUMN.get(row.get("producto"))
    if not column:
        return None
    value = row.get(column)
    return None if pd.isna(value) else float(value)


def cast_override_value(series: pd.Series, column: str, raw_value: Any) -> Any:
    if column == "periodo":
        return parse_period_value(raw_value)
    if column == "fecha_baja":
        return pd.to_datetime(raw_value, errors="coerce")
    if column in NUMERIC_COLUMNS or column in {"latitud", "longitud"}:
        return pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
    if column == "bandera":
        return canonical_bandera(raw_value)
    if column == "provincia":
        return canonical_provincia(raw_value)
    if column == "producto":
        return canonical_producto(raw_value)
    if column == "localidad":
        return canonical_localidad(raw_value)
    return raw_value


def load_overrides() -> pd.DataFrame:
    if not OVERRIDES_PATH.exists():
        return pd.DataFrame(
            columns=[
                "accion",
                "nro_inscripcion",
                "periodo",
                "producto",
                "campo",
                "valor_nuevo",
                "motivo",
                "fecha_agregado",
            ]
        )
    overrides = pd.read_csv(OVERRIDES_PATH)
    if overrides.empty:
        return overrides
    return overrides.fillna("")


def build_override_mask(df: pd.DataFrame, override_row: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    if str(override_row["nro_inscripcion"]).strip() not in {"", "*"}:
        mask &= df["nro_inscripcion"].astype("string") == str(override_row["nro_inscripcion"]).strip()

    periodo_value = str(override_row["periodo"]).strip()
    if periodo_value not in {"", "*"}:
        parsed_period = parse_period_value(periodo_value)
        mask &= df["periodo"] == parsed_period

    producto_value = str(override_row["producto"]).strip()
    if producto_value not in {"", "*"}:
        mask &= df["producto"] == canonical_producto(producto_value)

    return mask


def apply_overrides(
    df: pd.DataFrame,
    report_rows: list[dict[str, Any]],
    stage: str,
) -> tuple[pd.DataFrame, int]:
    overrides = load_overrides()
    if overrides.empty:
        return df, 0

    applied_count = 0
    for position, override_row in overrides.iterrows():
        motivo = str(override_row.get("motivo", "")).strip()
        if not motivo:
            raise ValueError(
                f"Override sin motivo en fila {position + 2} — todos los overrides requieren documentacion"
            )

        action = str(override_row.get("accion", "")).strip().lower()
        mask = build_override_mask(df, override_row)
        matched = df.loc[mask].copy()
        if matched.empty:
            print(f"WARNING: override en fila {position + 2} no matcheo ninguna fila")
            continue

        if action == "update" and stage == "update":
            field = snake_case(override_row.get("campo"))
            if field not in df.columns:
                raise ValueError(f"Override update apunta a campo inexistente: {field}")
            new_value = cast_override_value(df[field], field, override_row.get("valor_nuevo"))
            for index, row in matched.iterrows():
                old_value = row[field]
                df.at[index, field] = new_value
                report_rows.append(
                    build_cleaning_entry(
                        row=row,
                        motivo="override_update",
                        regla=motivo,
                        valor_problematico=f"{field}: {old_value} -> {new_value}",
                        accion="modified",
                    )
                )
            applied_count += len(matched)

        if action == "drop" and stage == "drop":
            for _, row in matched.iterrows():
                report_rows.append(
                    build_cleaning_entry(
                        row=row,
                        motivo="override_drop",
                        regla=motivo,
                        valor_problematico="fila eliminada por override",
                        accion="dropped",
                    )
                )
            df = df.loc[~mask].copy()
            applied_count += len(matched)

    return df, applied_count


def compute_iqr_outliers(df: pd.DataFrame, report_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, int]:
    to_drop = pd.Series(False, index=df.index)
    dropped = 0

    grouped = df.groupby(["producto", "periodo"], dropna=False)
    for (producto, periodo), group in grouped:
        if len(group) < 20:
            print(f"IQR skipped: grupo demasiado chico para {producto} / {periodo}")
            continue

        price_q1 = group["precio_surtidor"].quantile(0.25)
        price_q3 = group["precio_surtidor"].quantile(0.75)
        price_iqr = price_q3 - price_q1
        if pd.notna(price_iqr):
            low = price_q1 - 3 * price_iqr
            high = price_q3 + 3 * price_iqr
            mask = (group["precio_surtidor"] < low) | (group["precio_surtidor"] > high)
            for _, row in group.loc[mask].iterrows():
                report_rows.append(
                    build_cleaning_entry(
                        row=row,
                        motivo="outlier_iqr",
                        regla="outlier_precio_iqr",
                        valor_problematico=str(row["precio_surtidor"]),
                        accion="dropped",
                    )
                )
            to_drop.loc[group.loc[mask].index] = True

        volume_q1 = group["volumen"].quantile(0.25)
        volume_q3 = group["volumen"].quantile(0.75)
        volume_iqr = volume_q3 - volume_q1
        if pd.notna(volume_iqr):
            volume_high = volume_q3 + 5 * volume_iqr
            mask = (group["volumen"] > volume_high) & (group["volumen"] > 10000)
            mask &= group["volumen"] != 0
            for _, row in group.loc[mask].iterrows():
                report_rows.append(
                    build_cleaning_entry(
                        row=row,
                        motivo="outlier_iqr",
                        regla="outlier_volumen_iqr",
                        valor_problematico=str(row["volumen"]),
                        accion="dropped",
                    )
                )
            to_drop.loc[group.loc[mask].index] = True

    dropped = int(to_drop.sum())
    return df.loc[~to_drop].copy(), dropped


def preprocess() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos en {RAW_DATA_PATH}")

    CSV_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Leyendo Excel fuente...")
    raw_df = pd.read_excel(RAW_DATA_PATH, usecols=READ_COLUMNS)
    stats = CleaningStats(initial_rows=len(raw_df))
    report_rows: list[dict[str, Any]] = []

    print("Renombrando y tipando columnas...")
    raw_df = raw_df.rename(columns={column: RENAME_MAP.get(column, snake_case(column)) for column in raw_df.columns})
    df = raw_df.copy()

    df["periodo"] = df["periodo"].apply(parse_period_value)
    df["fecha_baja"] = pd.to_datetime(df["fecha_baja"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = parse_numeric(df[column])

    print("Normalizando coordenadas y catálogos...")
    df = build_coordinates(df)

    df["bandera"] = df["bandera"].apply(canonical_bandera)
    df["provincia"] = df["provincia"].apply(canonical_provincia)
    df["producto"] = df["producto"].apply(canonical_producto)
    df["localidad"] = df["localidad"].apply(canonical_localidad)
    df["direccion"] = df["direccion"].astype("string").str.strip()
    df["operador"] = df["operador"].astype("string").str.strip()
    df["tipo_negocio"] = df["tipo_negocio"].astype("string").str.strip()
    df["canal_comercializacion"] = df["canal_comercializacion"].astype("string").str.strip()
    df["cuit"] = df["cuit"].astype("string").str.strip()
    df["nro_inscripcion"] = df["nro_inscripcion"].astype("string").str.strip()

    print("Aplicando overrides de update...")
    df, stats.override_updates = apply_overrides(df, report_rows, stage="update")

    print("Ejecutando hard rules...")
    hard_rules = [
        ("null_precio_surtidor", df["precio_surtidor"].isna(), "precio_surtidor"),
        ("precio_surtidor_no_positivo", df["precio_surtidor"] <= 0, "precio_surtidor"),
        ("precio_con_impuestos_no_positivo", df["precio_con_impuestos"] <= 0, "precio_con_impuestos"),
        ("precio_sin_impuestos_negativo", df["precio_sin_impuestos"] < 0, "precio_sin_impuestos"),
        ("volumen_negativo", df["volumen"] < 0, "volumen"),
        ("nro_inscripcion_nulo", df["nro_inscripcion"].isna() | (df["nro_inscripcion"] == ""), "nro_inscripcion"),
        ("periodo_invalido", df["periodo"].isna(), "periodo"),
    ]

    for regla, mask, column in hard_rules:
        before = len(df)
        df = apply_drop_rule(df, mask.fillna(False), "hard_rule", regla, column, report_rows)
        stats.hard_drops += before - len(df)

    print("Ejecutando logic rules...")
    logic_rules = [
        (
            "precio_sin_mayor_que_con",
            df["precio_sin_impuestos"] > df["precio_con_impuestos"],
            "precio_sin_impuestos",
        ),
        (
            "surtidor_muy_bajo_vs_base",
            df["precio_surtidor"] < df["precio_sin_impuestos"] * 0.5,
            "precio_surtidor",
        ),
        (
            "surtidor_muy_alto_vs_con",
            df["precio_surtidor"] > df["precio_con_impuestos"] * 2.0,
            "precio_surtidor",
        ),
    ]

    for regla, mask, column in logic_rules:
        before = len(df)
        df = apply_drop_rule(df, mask.fillna(False), "logic_rule", regla, column, report_rows)
        stats.logic_drops += before - len(df)

    print("Saltando limpieza IQR avanzada para priorizar velocidad de demo...")
    stats.iqr_drops = 0

    print("Aplicando overrides de drop...")
    df, stats.override_drops = apply_overrides(df, report_rows, stage="drop")

    print("Aplicando filtro geográfico...")
    geo_mask = (
        df["latitud"].isna()
        | df["longitud"].isna()
        | (df["latitud"] < ARGENTINA_BBOX["lat_min"])
        | (df["latitud"] > ARGENTINA_BBOX["lat_max"])
        | (df["longitud"] < ARGENTINA_BBOX["lon_min"])
        | (df["longitud"] > ARGENTINA_BBOX["lon_max"])
    )
    before_geo = len(df)
    df = apply_drop_rule(df, geo_mask.fillna(False), "filtro_geografico", "coordenadas_fuera_de_rango", "geojson", report_rows)
    stats.geographic_drops = before_geo - len(df)

    df["activa_en_periodo"] = df["fecha_baja"].isna() | (df["fecha_baja"] > df["periodo"])
    df["carga_subnacional_informada_pct"] = (
        df[["tasa_vial", "tasa_municipal", "ingresos_brutos"]].fillna(0).sum(axis=1)
    )
    df["tiene_desglose_impositivo"] = df[
        [
            "impuesto_combustible_liquidos",
            "impuesto_dioxido_carbono",
            "tasa_vial",
            "tasa_municipal",
            "ingresos_brutos",
            "iva",
            "fondo_fiduciario_gnc",
        ]
    ].notna().any(axis=1)

    print("Adjuntando tablas de referencia...")
    df = attach_reference_tables(df)

    for column in STRING_CATEGORY_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("category")

    df["nro_inscripcion"] = df["nro_inscripcion"].astype("string")

    drop_columns = [column for column in ["latitud_raw", "longitud_raw"] if column in df.columns]
    df = df.drop(columns=drop_columns)

    print("Escribiendo outputs...")
    df.to_csv(CSV_DATASET_PATH, index=False)
    pd.DataFrame(report_rows).to_csv(CLEANING_REPORT_PATH, index=False)

    stats.final_rows = len(df)
    dataset_size_mb = CSV_DATASET_PATH.stat().st_size / (1024 * 1024)
    print("=== Resumen de limpieza ===")
    print(f"Filas iniciales:        {stats.initial_rows:>10,}")
    print(f"Drops por hard rules:   {stats.hard_drops:>10,}")
    print(f"Drops por logic rules:  {stats.logic_drops:>10,}")
    print(f"Drops por outliers IQR: {stats.iqr_drops:>10,}")
    print(f"Drops por overrides:    {stats.override_drops:>10,}")
    print(f"Updates por overrides:  {stats.override_updates:>10,}")
    print(f"Filtros geograficos:    {stats.geographic_drops:>10,}")
    print(f"Filas finales:          {stats.final_rows:>10,}")
    print(f"Tamaño dataset:         {dataset_size_mb:>10.1f} MB")
    print("==========================")


if __name__ == "__main__":
    preprocess()
