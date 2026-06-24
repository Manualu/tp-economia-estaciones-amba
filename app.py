from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

import queries
from config import AMBA_PROVINCES, BANDERA_COLORS, DEFAULT_PRODUCT, INITIAL_VIEW_STATE


st.set_page_config(page_title="Estaciones de servicio en AMBA", layout="wide")


PARTIDOS_GEOJSON_PATH = Path(__file__).resolve().parent / "data" / "processed" / "partidos_amba_ba.geojson"


MONTH_NAMES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}


def weighted_average_price(df: pd.DataFrame) -> float:
    valid = df[(df["volumen"] > 0) & df["precio_surtidor"].notna()]
    if valid.empty:
        return 0.0
    return float((valid["precio_surtidor"] * valid["volumen"]).sum() / valid["volumen"].sum())


def safe_number(value: object) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def safe_text(value: object, fallback: str = "Sin dato") -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def normalize_text(value: object) -> str:
    text = safe_text(value, fallback="").strip().lower()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def province_group(value: object) -> str:
    normalized = normalize_text(value)
    if normalized == "ciudad autonoma de buenos aires":
        return "caba"
    if normalized == "buenos aires":
        return "ba"
    return "other"


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    if "carga_subnacional_observable_pct" not in enriched.columns:
        source_columns = [column for column in ["tasa_vial", "tasa_municipal", "ingresos_brutos"] if column in enriched.columns]
        if source_columns:
            enriched["carga_subnacional_observable_pct"] = enriched[source_columns].sum(axis=1, min_count=1)
        else:
            enriched["carga_subnacional_observable_pct"] = pd.NA

    if "carga_subnacional_comparable" not in enriched.columns:
        if "tasa_vial_referencia_valor" in enriched.columns:
            enriched["carga_subnacional_comparable"] = enriched["carga_subnacional_observable_pct"].combine_first(
                enriched["tasa_vial_referencia_valor"]
            )
        else:
            enriched["carga_subnacional_comparable"] = enriched["carga_subnacional_observable_pct"]

    if "partido_amba" not in enriched.columns:
        enriched["partido_amba"] = pd.NA

    return enriched


def ensure_frontier_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    if "origen_menor_carga" not in enriched.columns:
        enriched["origen_menor_carga"] = enriched.get("provincia_menor_carga", pd.NA)

    if "origen_mayor_carga" not in enriched.columns:
        enriched["origen_mayor_carga"] = enriched.get("provincia_mayor_carga", pd.NA)

    if "tipo_frontera" not in enriched.columns:
        if {"provincia_menor_carga", "provincia_mayor_carga"}.issubset(enriched.columns):
            provinces = enriched[["provincia_menor_carga", "provincia_mayor_carga"]].astype("string")
            enriched["tipo_frontera"] = "Interpartido BA"
            caba_mask = (
                (provinces["provincia_menor_carga"] == "Ciudad Autonoma de Buenos Aires")
                | (provinces["provincia_mayor_carga"] == "Ciudad Autonoma de Buenos Aires")
            )
            enriched.loc[caba_mask, "tipo_frontera"] = "CABA vs Provincia"
        else:
            enriched["tipo_frontera"] = "Interpartido BA"

    if "partido_menor_carga" not in enriched.columns:
        enriched["partido_menor_carga"] = pd.NA
    if "partido_mayor_carga" not in enriched.columns:
        enriched["partido_mayor_carga"] = pd.NA

    return enriched.loc[enriched["origen_menor_carga"] != enriched["origen_mayor_carga"]].copy()


def enrich_visual_fields(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["color_rgba"] = enriched["bandera"].map(BANDERA_COLORS).apply(
        lambda color: color if isinstance(color, list) else BANDERA_COLORS["Otra"]
    )
    enriched["radio_metros"] = queries.calculate_radius(enriched["volumen"], radio_min_m=8, radio_max_m=290)
    enriched["tooltip_text"] = enriched.apply(
        lambda row: "\n".join(
            [
                safe_text(row.get("operador"), "Sin operador"),
                f"Bandera: {safe_text(row.get('bandera'))}",
                f"Localidad: {safe_text(row.get('localidad'))}",
                f"Precio surtidor: ${safe_number(row.get('precio_surtidor')):,.2f}",
                f"Variacion impositiva: {safe_number(row.get('variacion_impositiva')) * 100:,.2f}%",
                f"Volumen: {safe_number(row.get('volumen')):,.0f} m3",
            ]
        ),
        axis=1,
    )
    return enriched


def geometry_to_polygons(geometry: dict) -> list[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        return [coordinates[0]] if coordinates else []

    if geometry_type == "MultiPolygon":
        polygons: list[list[list[float]]] = []
        for polygon in coordinates:
            if polygon:
                polygons.append(polygon[0])
        return polygons

    return []


def load_partidos_geojson() -> dict | None:
    loader = getattr(queries, "load_partidos_amba_geojson", None)
    if callable(loader):
        return loader()

    path = PARTIDOS_GEOJSON_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_partido_boundary_data() -> list[dict[str, object]]:
    geojson = load_partidos_geojson()
    if geojson is None:
        return []

    rows: list[dict[str, object]] = []
    for feature in geojson.get("features", []):
        partido = feature.get("properties", {}).get("partido", "")
        for polygon in geometry_to_polygons(feature.get("geometry", {})):
            rows.append({"partido": partido, "polygon": polygon})
    return rows


def build_map(df: pd.DataFrame, show_partidos: bool) -> pdk.Deck:
    layers: list[pdk.Layer] = []

    if show_partidos:
        boundary_rows = build_partido_boundary_data()
        if boundary_rows:
            layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=boundary_rows,
                    get_polygon="polygon",
                    filled=False,
                    stroked=True,
                    get_line_color=[12, 18, 28, 235],
                    line_width_min_pixels=2,
                    line_width_max_pixels=3,
                    pickable=False,
                )
            )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["longitud", "latitud"],
            get_radius="radio_metros",
            get_fill_color="color_rgba",
            pickable=True,
            opacity=0.55,
            stroked=True,
            filled=True,
            line_width_min_pixels=1,
            get_line_color=[40, 40, 40, 170],
            radius_min_pixels=1,
            radius_max_pixels=28,
        )
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**INITIAL_VIEW_STATE),
        map_style="light",
        tooltip={
            "text": "{tooltip_text}",
            "style": {"backgroundColor": "white", "color": "black", "fontSize": "11px"},
        },
    )


def split_frontier_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = ensure_frontier_columns(df)
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

    caba_pairs = working.loc[caba_mask].copy()
    interpartido_pairs = working.loc[interpartido_mask].copy()

    if not caba_pairs.empty:
        caba_pairs["lado_menor_carga"] = caba_pairs["prov_group_menor"].map(
            {"caba": "CABA", "ba": "Provincia BA"}
        )
        caba_pairs["lado_mayor_carga"] = caba_pairs["prov_group_mayor"].map(
            {"caba": "CABA", "ba": "Provincia BA"}
        )

    return caba_pairs, interpartido_pairs


def reset_filters(defaults: dict[str, object]) -> None:
    for key, value in defaults.items():
        st.session_state[key] = value


def build_station_label(row: pd.Series) -> str:
    return (
        f"{row['nro_inscripcion']} | "
        f"{safe_text(row.get('operador'), 'Sin operador')} | "
        f"{safe_text(row.get('localidad'), 'Sin localidad')} | "
        f"{safe_text(row.get('bandera'))}"
    )


def render_color_legend(df: pd.DataFrame) -> None:
    entries = []
    for bandera in sorted(df["bandera"].dropna().unique().tolist()):
        color = BANDERA_COLORS.get(bandera, BANDERA_COLORS["Otra"])
        entries.append(
            f"<span style='display:inline-flex;align-items:center;margin-right:14px;margin-bottom:8px;'>"
            f"<span style='width:12px;height:12px;border-radius:50%;background-color:rgba({color[0]},{color[1]},{color[2]},{color[3]/255:.2f});display:inline-block;margin-right:6px;border:1px solid #444;'></span>"
            f"{bandera}</span>"
        )
    st.markdown("**Colores por bandera**", help="Cada color identifica la bandera comercial.")
    st.markdown("".join(entries), unsafe_allow_html=True)


def render_size_legend(df: pd.DataFrame) -> None:
    valid = df["volumen"].dropna()
    if valid.empty:
        return

    samples = []
    for quantile in (0.25, 0.5, 0.75):
        value = valid.quantile(quantile)
        if not math.isnan(value):
            samples.append(int(value))
    samples = sorted(set(samples))
    if not samples:
        return

    radii = queries.calculate_radius(pd.Series(samples), radio_min_m=8, radio_max_m=290)
    px_sizes = [max(7, min(32, int(radius / 7))) for radius in radii]
    items = []
    for volume, px_size in zip(samples, px_sizes):
        items.append(
            f"<div style='display:inline-flex;align-items:flex-end;margin-right:18px;'>"
            f"<div style='width:{px_size}px;height:{px_size}px;border-radius:50%;background:#9db7d5;border:1px solid #486581;opacity:.8;margin-right:8px;'></div>"
            f"<span>{volume:,.0f} m3</span></div>"
        )

    st.markdown("**Tamano de burbuja**", help="A mayor circulo, mayor volumen mensual del producto seleccionado.")
    st.markdown("".join(items), unsafe_allow_html=True)


filter_options = queries.get_filter_options()
periodos = [pd.Timestamp(period) for period in filter_options["periodos"]]
default_product = DEFAULT_PRODUCT if DEFAULT_PRODUCT in filter_options["productos"] else filter_options["productos"][0]
default_period = max(periodos)
years = sorted({periodo.year for periodo in periodos})
months_by_year = {year: sorted({periodo.month for periodo in periodos if periodo.year == year}) for year in years}

defaults = {
    "banderas": filter_options["banderas"],
    "producto": default_product,
    "periodo_year": default_period.year,
    "periodo_month": default_period.month,
    "provincias": [prov for prov in AMBA_PROVINCES if prov in filter_options["provincias"]],
    "localidades": [],
    "solo_activas": True,
    "mostrar_partidos": True,
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)

st.title("Mercado minorista de combustibles en AMBA")
st.caption(
    "Mapa exploratorio con foco en precio, volumen y carga impositiva observable. "
    "La superposicion opcional muestra partidos bonaerenses detectados en la base."
)

with st.sidebar:
    st.header("Filtros")
    st.multiselect("Bandera", options=filter_options["banderas"], key="banderas")
    st.selectbox("Producto", options=filter_options["productos"], key="producto")
    st.selectbox("Ano", options=years, key="periodo_year")

    month_options = months_by_year[st.session_state["periodo_year"]]
    if st.session_state["periodo_month"] not in month_options:
        st.session_state["periodo_month"] = month_options[-1]
    st.selectbox(
        "Mes",
        options=month_options,
        format_func=lambda month: MONTH_NAMES[month],
        key="periodo_month",
    )

    provincias = st.multiselect("Provincia", options=filter_options["provincias"], key="provincias")
    localidades_disponibles = queries.get_localidades(tuple(provincias))
    st.multiselect("Localidad", options=localidades_disponibles, key="localidades")
    st.toggle("Solo activas en el periodo seleccionado", key="solo_activas")
    st.toggle("Mostrar limites de partidos bonaerenses", key="mostrar_partidos")
    if st.button("Resetear filtros", use_container_width=True):
        reset_filters(defaults)
        st.rerun()

if not st.session_state["banderas"] or not st.session_state["provincias"]:
    st.warning("Selecciona al menos una bandera y una provincia para ver resultados.")
    st.stop()

selected_period = pd.Timestamp(
    year=st.session_state["periodo_year"],
    month=st.session_state["periodo_month"],
    day=1,
)

data = queries.query_estaciones(
    banderas=tuple(st.session_state["banderas"]),
    producto=st.session_state["producto"],
    periodo=selected_period,
    tipos_negocio=tuple(filter_options["tipos_negocio"]),
    provincias=tuple(st.session_state["provincias"]),
    localidades=tuple(st.session_state["localidades"]),
    solo_activas=bool(st.session_state["solo_activas"]),
)

hypothesis_data = queries.query_estaciones(
    banderas=tuple(st.session_state["banderas"]),
    producto=st.session_state["producto"],
    periodo=selected_period,
    tipos_negocio=tuple(filter_options["tipos_negocio"]),
    provincias=tuple([prov for prov in AMBA_PROVINCES if prov in filter_options["provincias"]]),
    localidades=tuple(),
    solo_activas=bool(st.session_state["solo_activas"]),
)

if data.empty:
    st.warning("No hay estaciones para esa combinacion de filtros.")
    st.stop()

data = ensure_derived_columns(data)
data = enrich_visual_fields(data)
hypothesis_data = ensure_derived_columns(hypothesis_data)

kpi_1, kpi_2, kpi_3 = st.columns(3)
kpi_1.metric("Estaciones visibles", f"{data['nro_inscripcion'].nunique():,}")
kpi_2.metric("Volumen total", f"{data['volumen'].sum():,.0f} m3")
kpi_3.metric("Precio surtidor promedio", f"${weighted_average_price(data):,.2f}")

tab_mapa, tab_hipotesis = st.tabs(["Mapa y detalle", "Hipotesis de frontera"])

with tab_mapa:
    st.pydeck_chart(build_map(data, show_partidos=bool(st.session_state["mostrar_partidos"])), use_container_width=True)

    legend_left, legend_right = st.columns([2, 1])
    with legend_left:
        render_color_legend(data)
    with legend_right:
        render_size_legend(data)

    chart_data = (
        data.groupby("bandera", observed=True)["volumen"]
        .sum()
        .sort_values(ascending=False)
        .rename("Volumen")
        .to_frame()
    )
    st.subheader("Volumen por bandera")
    st.bar_chart(chart_data)

    sorted_data = data.sort_values("volumen", ascending=False).reset_index(drop=True)
    station_labels = sorted_data.apply(build_station_label, axis=1)
    selected_label = st.selectbox("Ver detalle de una estacion", options=station_labels.tolist())
    selected_row = sorted_data.loc[station_labels == selected_label].iloc[0]

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.subheader("Detalle de estacion")
        st.write(f"**Operador:** {safe_text(selected_row.get('operador'), 'Sin operador')}")
        st.write(f"**Bandera:** {safe_text(selected_row.get('bandera'))}")
        st.write(f"**Direccion:** {safe_text(selected_row.get('direccion'))}")
        st.write(
            f"**Localidad / Provincia:** {safe_text(selected_row.get('localidad'))} / "
            f"{safe_text(selected_row.get('provincia'))}"
        )
        if pd.notna(selected_row.get("partido_amba")):
            st.write(f"**Partido bonaerense:** {selected_row['partido_amba']}")
        st.write(f"**Producto:** {safe_text(selected_row.get('producto'))}")
        st.write(f"**Volumen:** {safe_number(selected_row.get('volumen')):,.0f} m3")
        st.write(f"**Precio surtidor:** ${safe_number(selected_row.get('precio_surtidor')):,.2f}")
        st.write(f"**Precio sin impuestos:** ${safe_number(selected_row.get('precio_sin_impuestos')):,.2f}")
        st.write(f"**Precio con impuestos:** ${safe_number(selected_row.get('precio_con_impuestos')):,.2f}")
        st.write(f"**Variacion impositiva:** {safe_number(selected_row.get('variacion_impositiva')):,.4f}")

    with detail_right:
        st.subheader("Carga impositiva disponible")
        if bool(selected_row["tiene_desglose_impositivo"]):
            st.write(
                f"**Impuesto combustibles liquidos:** "
                f"{safe_number(selected_row.get('impuesto_combustible_liquidos')):,.2f}"
            )
            st.write(
                f"**Impuesto dioxido carbono:** "
                f"{safe_number(selected_row.get('impuesto_dioxido_carbono')):,.2f}"
            )
            st.write(f"**Tasa vial:** {safe_number(selected_row.get('tasa_vial')):,.2f}")
            st.write(f"**Tasa municipal:** {safe_number(selected_row.get('tasa_municipal')):,.2f}")
            st.write(f"**Ingresos brutos:** {safe_number(selected_row.get('ingresos_brutos')):,.2f}")
            st.write(f"**IVA:** {safe_number(selected_row.get('iva')):,.2f}")
            st.write(f"**Fondo fiduciario GNC:** {safe_number(selected_row.get('fondo_fiduciario_gnc')):,.2f}")
            st.write(
                f"**Carga subnacional observable:** "
                f"{safe_number(selected_row.get('carga_subnacional_observable_pct')):,.2f}"
            )
        else:
            st.info("Esta estacion no trae desglose impositivo completo en el dataset seleccionado.")

        if pd.notna(selected_row.get("iibb_referencia_texto")):
            st.write(
                f"**IIBB de referencia ({safe_text(selected_row.get('provincia'))}):** "
                f"{selected_row['iibb_referencia_texto']}"
            )
        if pd.notna(selected_row.get("tasa_vial_referencia_valor")):
            st.write(
                f"**Tasa vial de referencia por localidad:** "
                f"{safe_number(selected_row.get('tasa_vial_referencia_valor')):,.3f}"
            )

    st.subheader("Estaciones visibles")
    st.dataframe(
        sorted_data[
            [
                "operador",
                "bandera",
                "localidad",
                "provincia",
                "partido_amba",
                "volumen",
                "precio_surtidor",
                "precio_sin_impuestos",
                "precio_con_impuestos",
                "variacion_impositiva",
                "carga_subnacional_comparable",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_hipotesis:
    st.subheader("Proxy de efecto frontera")
    st.caption(
        "La comparacion se separa entre CABA vs Provincia de Buenos Aires y el caso interpartido bonaerense. "
        "Para no perder pares de frontera por recortes geograficos del mapa, esta pestana evalua AMBA completo "
        "con el producto, periodo, banderas y estado de actividad seleccionados."
    )

    frontier_pairs = queries.compute_frontier_pairs(hypothesis_data)
    if frontier_pairs.empty:
        st.info("No se encontraron pares cercanos con carga subnacional comparable para esta combinacion de filtros.")
    else:
        frontier_pairs = ensure_frontier_columns(frontier_pairs)
        caba_pairs, interpartido_pairs = split_frontier_pairs(frontier_pairs)
        if caba_pairs.empty and interpartido_pairs.empty:
            st.info("No se pudieron construir pares validos para CABA vs Provincia o Interpartido BA.")
            st.stop()

        st.write(
            "Cada par compara estaciones cercanas. Si la estacion con menor carga subnacional comparable tambien vende "
            "mas volumen, el par se considera consistente con la hipotesis."
        )

        tab_caba, tab_partidos = st.tabs(["CABA vs Provincia", "Interpartido BA"])

        with tab_caba:
            if caba_pairs.empty:
                st.info("No hay pares validos de CABA vs Provincia con estos filtros.")
            else:
                k1, k2, k3 = st.columns(3)
                k1.metric("Pares cercanos", f"{len(caba_pairs):,}")
                k2.metric("Distancia promedio", f"{caba_pairs['distancia_km'].mean():.2f} km")
                k3.metric("Favorece hipotesis", f"{100 * caba_pairs['favorece_hipotesis'].mean():.1f}%")

                st.dataframe(
                    caba_pairs[
                        [
                            "distancia_km",
                            "lado_menor_carga",
                            "origen_menor_carga",
                            "estacion_menor_carga",
                            "lado_mayor_carga",
                            "origen_mayor_carga",
                            "estacion_mayor_carga",
                            "carga_menor_pct",
                            "carga_mayor_pct",
                            "volumen_menor_carga",
                            "volumen_mayor_carga",
                            "precio_menor_carga",
                            "precio_mayor_carga",
                            "favorece_hipotesis",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        with tab_partidos:
            if interpartido_pairs.empty:
                st.info("No hay pares validos interpartido BA con estos filtros.")
            else:
                k1, k2, k3 = st.columns(3)
                k1.metric("Pares cercanos", f"{len(interpartido_pairs):,}")
                k2.metric("Distancia promedio", f"{interpartido_pairs['distancia_km'].mean():.2f} km")
                k3.metric("Favorece hipotesis", f"{100 * interpartido_pairs['favorece_hipotesis'].mean():.1f}%")

                st.dataframe(
                    interpartido_pairs[
                        [
                            "distancia_km",
                            "partido_menor_carga",
                            "estacion_menor_carga",
                            "partido_mayor_carga",
                            "estacion_mayor_carga",
                            "carga_menor_pct",
                            "carga_mayor_pct",
                            "volumen_menor_carga",
                            "volumen_mayor_carga",
                            "precio_menor_carga",
                            "precio_mayor_carga",
                            "favorece_hipotesis",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
