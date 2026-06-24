from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

import queries
from config import AMBA_PROVINCES, DATASET_PATH, DEFAULT_PRODUCT, PROCESSED_DIR


OUTPUT_DIR = PROCESSED_DIR / "presentacion_final"
COLOR_PRIMARY = "#2C2C2A"
COLOR_ACCENT = "#BA7517"
COLOR_BACKGROUND = "#F7F5EE"


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def slugify(value: str) -> str:
    normalized = normalize_text(value)
    normalized = normalized.replace(" ", "-").replace("/", "-")
    return "".join(char for char in normalized if char.isalnum() or char == "-")


def province_group(value: object) -> str:
    normalized = normalize_text(value)
    if normalized == "ciudad autonoma de buenos aires":
        return "caba"
    if normalized == "buenos aires":
        return "ba"
    return "other"


def first_half_2025_best_period(producto: str) -> pd.Timestamp:
    df = pd.read_csv(DATASET_PATH, parse_dates=["periodo"], low_memory=False)
    filtered = df[
        (df["periodo"] >= "2025-01-01")
        & (df["periodo"] < "2025-07-01")
        & (df["producto"] == producto)
    ].copy()
    if filtered.empty:
        raise ValueError("No hay datos para el producto elegido en el primer semestre de 2025.")

    summary = (
        filtered.groupby("periodo", dropna=False)
        .agg(
            filas=("periodo", "size"),
            estaciones=("nro_inscripcion", "nunique"),
            con_desglose=("tiene_desglose_impositivo", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            con_variacion=("variacion_impositiva", lambda s: int(pd.Series(s).notna().sum())),
        )
        .reset_index()
        .sort_values(["estaciones", "filas", "periodo"], ascending=[False, False, True])
    )
    return pd.Timestamp(summary.iloc[0]["periodo"])


def load_period_data(producto: str, periodo: pd.Timestamp) -> pd.DataFrame:
    filter_options = queries.get_filter_options()
    provincias = tuple(prov for prov in filter_options["provincias"] if province_group(prov) in {"caba", "ba"})
    return queries.query_estaciones(
        banderas=tuple(filter_options["banderas"]),
        producto=producto,
        periodo=periodo,
        provincias=provincias,
        localidades=tuple(),
        solo_activas=True,
        tipos_negocio=tuple(filter_options["tipos_negocio"]),
    )


def enrich_frontier_pairs(frontier_pairs: pd.DataFrame) -> pd.DataFrame:
    working = frontier_pairs.copy()
    working["prov_group_menor"] = working["provincia_menor_carga"].apply(province_group)
    working["prov_group_mayor"] = working["provincia_mayor_carga"].apply(province_group)

    caba_mask = (
        ((working["prov_group_menor"] == "caba") & (working["prov_group_mayor"] == "ba"))
        | ((working["prov_group_menor"] == "ba") & (working["prov_group_mayor"] == "caba"))
    )
    interpartido_mask = (
        (working["prov_group_menor"] == "ba")
        & (working["prov_group_mayor"] == "ba")
        & working["partido_menor_carga"].notna()
        & working["partido_mayor_carga"].notna()
        & (working["partido_menor_carga"] != working["partido_mayor_carga"])
    )

    working["grupo_presentacion"] = pd.NA
    working.loc[caba_mask, "grupo_presentacion"] = "CABA vs Provincia"
    working.loc[interpartido_mask, "grupo_presentacion"] = "Interpartido BA"
    working = working.loc[working["grupo_presentacion"].notna()].copy()

    working["jurisdiccion_menor_carga"] = working.apply(
        lambda row: "CABA"
        if row["prov_group_menor"] == "caba"
        else row["partido_menor_carga"]
        if pd.notna(row["partido_menor_carga"])
        else row["origen_menor_carga"],
        axis=1,
    )
    working["jurisdiccion_mayor_carga"] = working.apply(
        lambda row: "CABA"
        if row["prov_group_mayor"] == "caba"
        else row["partido_mayor_carga"]
        if pd.notna(row["partido_mayor_carga"])
        else row["origen_mayor_carga"],
        axis=1,
    )
    working["gap_volumen_litros"] = working["gap_volumen"] * 1000.0
    return working


def aggregate_map_pairs(frontier_pairs: pd.DataFrame) -> pd.DataFrame:
    working = frontier_pairs.copy()
    working["jurisdiccion_1"] = working.apply(
        lambda row: "CABA"
        if row["grupo_presentacion"] == "CABA vs Provincia"
        else min(str(row["jurisdiccion_menor_carga"]), str(row["jurisdiccion_mayor_carga"])),
        axis=1,
    )
    working["jurisdiccion_2"] = working.apply(
        lambda row: str(row["jurisdiccion_mayor_carga"])
        if row["grupo_presentacion"] == "CABA vs Provincia" and str(row["jurisdiccion_menor_carga"]) == "CABA"
        else str(row["jurisdiccion_menor_carga"])
        if row["grupo_presentacion"] == "CABA vs Provincia" and str(row["jurisdiccion_mayor_carga"]) == "CABA"
        else max(str(row["jurisdiccion_menor_carga"]), str(row["jurisdiccion_mayor_carga"])),
        axis=1,
    )
    working["menor_carga_en_par"] = working["jurisdiccion_menor_carga"]
    working["distancia_km"] = working["distancia_km"].astype(float)
    working["gap_carga_pct_abs"] = working["gap_carga_pct"].abs()

    grouped = (
        working.groupby(["grupo_presentacion", "jurisdiccion_1", "jurisdiccion_2"], dropna=False)
        .agg(
            pares_estaciones=("grupo_presentacion", "size"),
            distancia_promedio_km=("distancia_km", "mean"),
            distancia_mediana_km=("distancia_km", "median"),
            gap_carga_promedio_pct=("gap_carga_pct", "mean"),
            gap_carga_mediana_pct=("gap_carga_pct", "median"),
            gap_carga_mediana_abs_pct=("gap_carga_pct_abs", "median"),
            gap_volumen_promedio_litros=("gap_volumen_litros", "mean"),
            gap_volumen_mediano_litros=("gap_volumen_litros", "median"),
            consistencia_hipotesis_pct=("favorece_hipotesis", lambda s: 100.0 * float(pd.Series(s).mean())),
        )
        .reset_index()
    )

    menor_frecuente = (
        working.groupby(["grupo_presentacion", "jurisdiccion_1", "jurisdiccion_2", "menor_carga_en_par"], dropna=False)
        .size()
        .reset_index(name="conteo")
        .sort_values(
            ["grupo_presentacion", "jurisdiccion_1", "jurisdiccion_2", "conteo", "menor_carga_en_par"],
            ascending=[True, True, True, False, True],
        )
        .drop_duplicates(["grupo_presentacion", "jurisdiccion_1", "jurisdiccion_2"])
        .rename(columns={"menor_carga_en_par": "jurisdiccion_con_menor_carga_mas_frecuente"})
    )

    grouped = grouped.merge(
        menor_frecuente[
            [
                "grupo_presentacion",
                "jurisdiccion_1",
                "jurisdiccion_2",
                "jurisdiccion_con_menor_carga_mas_frecuente",
            ]
        ],
        on=["grupo_presentacion", "jurisdiccion_1", "jurisdiccion_2"],
        how="left",
    )
    grouped["color_principal_hex"] = COLOR_PRIMARY
    grouped["color_acento_hex"] = COLOR_ACCENT
    grouped["color_fondo_hex"] = COLOR_BACKGROUND
    grouped["color_visual_hex"] = grouped["consistencia_hipotesis_pct"].apply(
        lambda value: COLOR_ACCENT if float(value) >= 50.0 else COLOR_PRIMARY
    )
    return grouped.sort_values(
        ["grupo_presentacion", "pares_estaciones", "consistencia_hipotesis_pct", "distancia_promedio_km"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def select_map_examples(map_pairs: pd.DataFrame) -> pd.DataFrame:
    cleaned = map_pairs.loc[map_pairs["gap_carga_mediana_abs_pct"] <= 200].copy()
    cleaned = cleaned.loc[
        ~cleaned["jurisdiccion_1"].astype(str).str.startswith("Localidad:")
        & ~cleaned["jurisdiccion_2"].astype(str).str.startswith("Localidad:")
    ].copy()

    caba_examples = (
        cleaned.loc[cleaned["grupo_presentacion"] == "CABA vs Provincia"]
        .sort_values(["pares_estaciones", "consistencia_hipotesis_pct", "distancia_promedio_km"], ascending=[False, False, True])
        .head(6)
        .copy()
    )
    caba_examples["escenario_slide"] = "CABA vs Provincia"

    inter_examples = (
        cleaned.loc[cleaned["grupo_presentacion"] == "Interpartido BA"]
        .sort_values(["pares_estaciones", "consistencia_hipotesis_pct", "distancia_promedio_km"], ascending=[False, False, True])
        .head(6)
        .copy()
    )
    inter_examples["escenario_slide"] = "Municipio vs Municipio"

    dos_municipios = (
        cleaned.loc[cleaned["grupo_presentacion"] == "Interpartido BA"]
        .sort_values(
            ["consistencia_hipotesis_pct", "pares_estaciones", "gap_carga_mediana_abs_pct", "distancia_promedio_km"],
            ascending=[False, False, False, True],
        )
        .head(2)
        .copy()
    )
    dos_municipios["escenario_slide"] = "Dos municipios destacados"

    return pd.concat([caba_examples, inter_examples, dos_municipios], ignore_index=True)


def build_scatter_dataset(estaciones: pd.DataFrame) -> pd.DataFrame:
    working = estaciones.copy()
    working["jurisdiccion"] = working.apply(
        lambda row: "CABA"
        if province_group(row["provincia"]) == "caba"
        else row["partido_amba"]
        if pd.notna(row["partido_amba"])
        else row["localidad"],
        axis=1,
    )
    working["grupo_jurisdiccion"] = working["provincia"].apply(
        lambda value: "CABA" if province_group(value) == "caba" else "Partido BA"
    )
    working["tasa_vial_usable_pct"] = working["tasa_vial_referencia_valor"].combine_first(working["tasa_vial"]).fillna(0.0)

    scatter = (
        working.groupby(["grupo_jurisdiccion", "jurisdiccion"], dropna=False)
        .agg(
            estaciones=("nro_inscripcion", "nunique"),
            volumen_m3_mes=("volumen", "sum"),
            precio_promedio=("precio_surtidor", "mean"),
            variacion_impositiva_promedio=("variacion_impositiva", "mean"),
            tasa_vial_media_pct=("tasa_vial_usable_pct", "mean"),
            carga_subnacional_media_pct=("carga_subnacional_comparable", "mean"),
        )
        .reset_index()
    )
    scatter["litros_mes"] = scatter["volumen_m3_mes"] * 1000.0
    tasa_vial_media_amba = float(scatter["tasa_vial_media_pct"].mean())
    scatter["tasa_vial_media_amba_pct"] = tasa_vial_media_amba
    scatter["desvio_tasa_vial_pp"] = scatter["tasa_vial_media_pct"] - tasa_vial_media_amba
    if abs(tasa_vial_media_amba) < 1e-9:
        scatter["desvio_tasa_vial_pct"] = 0.0
    else:
        scatter["desvio_tasa_vial_pct"] = 100.0 * scatter["desvio_tasa_vial_pp"] / tasa_vial_media_amba
    scatter["color_principal_hex"] = COLOR_PRIMARY
    scatter["color_acento_hex"] = COLOR_ACCENT
    scatter["color_fondo_hex"] = COLOR_BACKGROUND
    scatter["color_visual_hex"] = scatter["grupo_jurisdiccion"].map(
        {"CABA": COLOR_ACCENT, "Partido BA": COLOR_PRIMARY}
    ).fillna(COLOR_PRIMARY)

    return scatter.sort_values(["grupo_jurisdiccion", "litros_mes"], ascending=[True, False]).reset_index(drop=True)


def export_metadata(periodo: pd.Timestamp, producto: str, estaciones: pd.DataFrame, output_prefix: str) -> Path:
    metadata = pd.DataFrame(
        [
            {
                "periodo_seleccionado": periodo.strftime("%Y-%m-%d"),
                "producto": producto,
                "provincias_incluidas": ", ".join(AMBA_PROVINCES),
                "criterio_periodo": "Mes de enero-junio 2025 con mayor cantidad de estaciones para el producto elegido.",
                "estaciones_analizadas": int(estaciones["nro_inscripcion"].nunique()),
                "color_principal_hex": COLOR_PRIMARY,
                "color_acento_hex": COLOR_ACCENT,
                "color_fondo_hex": COLOR_BACKGROUND,
            }
        ]
    )
    path = OUTPUT_DIR / f"{output_prefix}_metadata.csv"
    metadata.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta CSVs para la presentacion final del TP de economia.")
    parser.add_argument("--producto", default=DEFAULT_PRODUCT, help="Producto a analizar. Por defecto usa el de la app.")
    parser.add_argument(
        "--periodo",
        default="auto",
        help="Periodo YYYY-MM o YYYY-MM-DD. Si se usa 'auto', toma el mejor mes de enero-junio de 2025.",
    )
    args = parser.parse_args()

    producto = args.producto
    periodo = (
        first_half_2025_best_period(producto)
        if args.periodo == "auto"
        else pd.Timestamp(args.periodo if len(args.periodo) > 7 else f"{args.periodo}-01")
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    estaciones = load_period_data(producto=producto, periodo=periodo)
    frontier_pairs = enrich_frontier_pairs(queries.compute_frontier_pairs(estaciones))
    map_pairs = aggregate_map_pairs(frontier_pairs)
    map_examples = select_map_examples(map_pairs)
    scatter = build_scatter_dataset(estaciones)

    output_prefix = f"presentacion_{periodo.strftime('%Y-%m')}_{slugify(producto)}"

    metadata_path = export_metadata(periodo=periodo, producto=producto, estaciones=estaciones, output_prefix=output_prefix)
    frontier_pairs_path = OUTPUT_DIR / f"{output_prefix}_pares_frontera_estaciones.csv"
    map_pairs_path = OUTPUT_DIR / f"{output_prefix}_mapa_pares_agregados.csv"
    map_examples_path = OUTPUT_DIR / f"{output_prefix}_mapa_ejemplos.csv"
    scatter_path = OUTPUT_DIR / f"{output_prefix}_scatter_jurisdicciones.csv"

    frontier_pairs.to_csv(frontier_pairs_path, index=False, encoding="utf-8-sig")
    map_pairs.to_csv(map_pairs_path, index=False, encoding="utf-8-sig")
    map_examples.to_csv(map_examples_path, index=False, encoding="utf-8-sig")
    scatter.to_csv(scatter_path, index=False, encoding="utf-8-sig")

    print(f"Periodo seleccionado: {periodo.strftime('%Y-%m-%d')}")
    print(f"Producto: {producto}")
    print("Archivos exportados:")
    print(f"- {metadata_path}")
    print(f"- {frontier_pairs_path}")
    print(f"- {map_pairs_path}")
    print(f"- {map_examples_path}")
    print(f"- {scatter_path}")


if __name__ == "__main__":
    main()
