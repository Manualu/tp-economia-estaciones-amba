from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import queries
from config import DEFAULT_PRODUCT, PARTIDOS_AMBA_PATH
from export_presentacion_final import (
    COLOR_ACCENT,
    COLOR_BACKGROUND,
    COLOR_PRIMARY,
    OUTPUT_DIR,
    aggregate_map_pairs,
    enrich_frontier_pairs,
    first_half_2025_best_period,
    load_period_data,
    select_map_examples,
    slugify,
)


WIDTH = 1800
HEIGHT = 1000


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def mix_with_background(hex_color: str, alpha: float) -> tuple[int, int, int]:
    base = np.array(hex_to_rgb(hex_color), dtype=float)
    bg = np.array(hex_to_rgb(COLOR_BACKGROUND), dtype=float)
    mixed = alpha * base + (1 - alpha) * bg
    return tuple(int(round(v)) for v in mixed)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_spacing: int = 6,
) -> None:
    x, y = position
    for line in text.splitlines():
        draw.text((x, y), line, fill=fill, font=font)
        _, h = text_size(draw, line, font)
        y += h + line_spacing


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    w, h = text_size(draw, text, font)
    draw.text((center[0] - w / 2, center[1] - h / 2), text, fill=fill, font=font)


def load_partido_geometry() -> tuple[list[dict], dict[str, tuple[float, float]]]:
    geojson = queries.load_partidos_amba_geojson()
    if geojson is None:
        raise FileNotFoundError(f"No se encontro el GeoJSON de partidos en {PARTIDOS_AMBA_PATH}")

    features = geojson.get("features", [])
    centroids: dict[str, tuple[float, float]] = {}
    for feature in features:
        properties = feature.get("properties", {})
        partido = properties.get("partido")
        bbox = properties.get("bbox")
        if partido and bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            centroids[str(partido)] = ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)
    return features, centroids


def geometry_outer_rings(geometry: dict) -> list[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        return [coordinates[0]] if coordinates else []
    if geometry_type == "MultiPolygon":
        return [polygon[0] for polygon in coordinates if polygon]
    return []


def make_transform(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    box: tuple[int, int, int, int],
):
    left, top, right, bottom = box
    width = right - left
    height = bottom - top

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = left + (x - min_x) / (max_x - min_x) * width
        py = bottom - (y - min_y) / (max_y - min_y) * height
        return px, py

    return transform


def build_map_visual(
    estaciones: pd.DataFrame,
    map_examples: pd.DataFrame,
    output_png: Path,
    periodo: pd.Timestamp,
    producto: str,
) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), hex_to_rgb(COLOR_BACKGROUND))
    draw = ImageDraw.Draw(image)
    title_font = load_font(42, bold=True)
    subtitle_font = load_font(19)
    label_font = load_font(15)
    small_font = load_font(13)
    number_font = load_font(15, bold=True)

    map_box = (70, 150, 1180, 760)
    notes_box = (1230, 150, 1730, 890)
    legend_box = (70, 810, 1180, 940)

    features, centroids = load_partido_geometry()
    caba_rows = estaciones.loc[estaciones["provincia"].astype(str).str.contains("Ciudad", na=False)].copy()
    caba_center = (float(caba_rows["longitud"].mean()), float(caba_rows["latitud"].mean()))

    selected_jurisdictions = set(map_examples["jurisdiccion_1"].astype(str)).union(set(map_examples["jurisdiccion_2"].astype(str)))
    selected_jurisdictions.discard("CABA")

    caba_examples = map_examples.loc[map_examples["grupo_presentacion"] == "CABA vs Provincia"].head(4)
    inter_examples = map_examples.loc[map_examples["grupo_presentacion"] == "Interpartido BA"].head(4)
    plot_rows = pd.concat([caba_examples, inter_examples], ignore_index=True).reset_index(drop=True).copy()
    plot_rows["line_number"] = plot_rows.index + 1
    plot_rows["x1"] = plot_rows["jurisdiccion_1"].map(lambda name: caba_center[0] if name == "CABA" else centroids.get(str(name), (None, None))[0])
    plot_rows["y1"] = plot_rows["jurisdiccion_1"].map(lambda name: caba_center[1] if name == "CABA" else centroids.get(str(name), (None, None))[1])
    plot_rows["x2"] = plot_rows["jurisdiccion_2"].map(lambda name: caba_center[0] if name == "CABA" else centroids.get(str(name), (None, None))[0])
    plot_rows["y2"] = plot_rows["jurisdiccion_2"].map(lambda name: caba_center[1] if name == "CABA" else centroids.get(str(name), (None, None))[1])
    plot_rows = plot_rows.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

    xs = pd.concat([plot_rows["x1"], plot_rows["x2"]], ignore_index=True)
    ys = pd.concat([plot_rows["y1"], plot_rows["y2"]], ignore_index=True)
    min_x, max_x = float(xs.min() - 0.12), float(xs.max() + 0.12)
    min_y, max_y = float(ys.min() - 0.09), float(ys.max() + 0.08)
    transform = make_transform(min_x, max_x, min_y, max_y, map_box)

    draw.rounded_rectangle(map_box, radius=22, outline=mix_with_background(COLOR_PRIMARY, 0.22), width=2, fill=hex_to_rgb(COLOR_BACKGROUND))
    draw.rounded_rectangle(notes_box, radius=22, outline=mix_with_background(COLOR_PRIMARY, 0.22), width=2, fill=mix_with_background(COLOR_PRIMARY, 0.03))
    draw.rounded_rectangle(legend_box, radius=22, outline=mix_with_background(COLOR_PRIMARY, 0.18), width=1, fill=mix_with_background(COLOR_PRIMARY, 0.02))

    for feature in features:
        partido = str(feature.get("properties", {}).get("partido", ""))
        bbox = feature.get("properties", {}).get("bbox")
        if bbox:
            f_min_lon, f_min_lat, f_max_lon, f_max_lat = bbox
            if f_max_lon < min_x or f_min_lon > max_x or f_max_lat < min_y or f_min_lat > max_y:
                continue
        is_selected = partido in selected_jurisdictions
        fill = mix_with_background(COLOR_ACCENT if is_selected else COLOR_PRIMARY, 0.12 if is_selected else 0.03)
        edge = mix_with_background(COLOR_PRIMARY, 0.48 if is_selected else 0.16)
        for ring in geometry_outer_rings(feature.get("geometry", {})):
            points = [transform((lon, lat)) for lon, lat in ring]
            draw.polygon(points, fill=fill, outline=edge)

    for row in plot_rows.itertuples(index=False):
        line_color = hex_to_rgb(COLOR_ACCENT if float(row.consistencia_hipotesis_pct) >= 50 else COLOR_PRIMARY)
        width = int(round(2 + min(float(row.pares_estaciones) / 10.0, 5)))
        p1 = transform((float(row.x1), float(row.y1)))
        p2 = transform((float(row.x2), float(row.y2)))
        draw.line([p1, p2], fill=line_color, width=width)
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        r = 12
        draw.ellipse((mid[0] - r, mid[1] - r, mid[0] + r, mid[1] + r), fill=hex_to_rgb(COLOR_BACKGROUND), outline=line_color, width=2)
        draw_centered_text(draw, mid, str(int(row.line_number)), number_font, line_color)

    for name in sorted(selected_jurisdictions):
        if name not in centroids:
            continue
        x, y = transform(centroids[name])
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=hex_to_rgb(COLOR_PRIMARY))
        draw.text((x + 6, y - 8), name, fill=hex_to_rgb(COLOR_PRIMARY), font=label_font)

    caba_px = transform(caba_center)
    draw.ellipse((caba_px[0] - 8, caba_px[1] - 8, caba_px[0] + 8, caba_px[1] + 8), fill=hex_to_rgb(COLOR_ACCENT), outline=hex_to_rgb(COLOR_PRIMARY), width=2)
    draw.text((caba_px[0] + 8, caba_px[1] - 12), "CABA", fill=hex_to_rgb(COLOR_PRIMARY), font=load_font(18, bold=True))

    draw.text((70, 46), "Lectura espacial de comparaciones de frontera", fill=hex_to_rgb(COLOR_PRIMARY), font=title_font)
    draw.text(
        (70, 100),
        f"{producto} | {periodo.strftime('%Y-%m')} | Ejemplos de CABA vs Provincia e Interpartido BA",
        fill=hex_to_rgb(COLOR_PRIMARY),
        font=subtitle_font,
    )

    draw.text((1255, 175), "Ejemplos seleccionados", fill=hex_to_rgb(COLOR_PRIMARY), font=load_font(24, bold=True))
    note_y = 225
    for row in plot_rows.itertuples(index=False):
        line_color = COLOR_ACCENT if float(row.consistencia_hipotesis_pct) >= 50 else COLOR_PRIMARY
        block = (
            f"{int(row.line_number)}. {row.jurisdiccion_1} vs {row.jurisdiccion_2}\n"
            f"Pares: {int(row.pares_estaciones)} | Distancia media: {float(row.distancia_promedio_km):.2f} km\n"
            f"Consistencia: {float(row.consistencia_hipotesis_pct):.0f}% | Menor carga frecuente: {row.jurisdiccion_con_menor_carga_mas_frecuente}"
        )
        draw.rounded_rectangle((1250, note_y - 10, 1710, note_y + 66), radius=14, fill=mix_with_background(line_color, 0.06), outline=mix_with_background(line_color, 0.35), width=2)
        draw_multiline(draw, (1268, note_y), block, small_font, hex_to_rgb(COLOR_PRIMARY), line_spacing=2)
        note_y += 94

    legend_text = (
        "Color ámbar: mayoría de comparaciones consistente con la hipótesis\n"
        "Color grafito: mayoría de comparaciones no consistente\n"
        "Grosor de línea: más pares comparados entre jurisdicciones"
    )
    draw_multiline(draw, (95, 840), legend_text, label_font, hex_to_rgb(COLOR_PRIMARY), line_spacing=5)

    image.save(output_png)


def choose_scatter_labels(scatter: pd.DataFrame) -> pd.DataFrame:
    x_column = "desvio_carga_pp" if "desvio_carga_pp" in scatter.columns else "desvio_tasa_vial_pct"
    top_volume = scatter.sort_values("litros_mes", ascending=False).head(8)
    extremes_left = scatter.nsmallest(3, x_column)
    extremes_right = scatter.nlargest(3, x_column)
    caba = scatter.loc[scatter["grupo_jurisdiccion"] == "CABA"] if "grupo_jurisdiccion" in scatter.columns else scatter.loc[scatter["jurisdiccion"] == "CABA"]
    labels = pd.concat([top_volume, extremes_left, extremes_right, caba], ignore_index=True)
    dedupe_column = "estacion" if "estacion" in labels.columns else "jurisdiccion"
    return labels.drop_duplicates(subset=[dedupe_column]).copy()


def build_caba_prov_scatter_dataset(frontier_pairs: pd.DataFrame) -> pd.DataFrame:
    caba_pairs = frontier_pairs.loc[frontier_pairs["grupo_presentacion"] == "CABA vs Provincia"].copy()
    records: list[dict[str, object]] = []

    for row in caba_pairs.itertuples(index=False):
        records.append(
            {
                "estacion": row.estacion_menor_carga,
                "grupo_jurisdiccion": "CABA" if "Ciudad Autónoma de Buenos Aires" in str(row.provincia_menor_carga) else "Provincia BA",
                "jurisdiccion": "CABA" if "Ciudad Autónoma de Buenos Aires" in str(row.provincia_menor_carga) else row.partido_menor_carga or row.origen_menor_carga,
                "carga_pct": float(row.carga_menor_pct),
                "litros_mes": float(row.volumen_menor_carga) * 1000.0,
            }
        )
        records.append(
            {
                "estacion": row.estacion_mayor_carga,
                "grupo_jurisdiccion": "CABA" if "Ciudad Autónoma de Buenos Aires" in str(row.provincia_mayor_carga) else "Provincia BA",
                "jurisdiccion": "CABA" if "Ciudad Autónoma de Buenos Aires" in str(row.provincia_mayor_carga) else row.partido_mayor_carga or row.origen_mayor_carga,
                "carga_pct": float(row.carga_mayor_pct),
                "litros_mes": float(row.volumen_mayor_carga) * 1000.0,
            }
        )

    scatter = pd.DataFrame(records).drop_duplicates(subset=["estacion", "grupo_jurisdiccion", "carga_pct", "litros_mes"]).copy()
    carga_media = float(scatter["carga_pct"].mean())
    scatter["desvio_carga_pp"] = scatter["carga_pct"] - carga_media
    return scatter.sort_values(["grupo_jurisdiccion", "litros_mes"], ascending=[True, False]).reset_index(drop=True)


def build_scatter_visual(scatter: pd.DataFrame, output_png: Path, periodo: pd.Timestamp, producto: str) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), hex_to_rgb(COLOR_BACKGROUND))
    draw = ImageDraw.Draw(image)
    title_font = load_font(42, bold=True)
    subtitle_font = load_font(19)
    axis_font = load_font(18)
    tick_font = load_font(14)
    label_font = load_font(13)

    draw.text((90, 50), "Litros por mes vs carga subnacional", fill=hex_to_rgb(COLOR_PRIMARY), font=title_font)
    draw.text(
        (90, 105),
        f"{producto} | {periodo.strftime('%Y-%m')} | Solo estaciones incluidas en comparaciones CABA vs Provincia",
        fill=hex_to_rgb(COLOR_PRIMARY),
        font=subtitle_font,
    )

    plot_box = (140, 170, 1670, 860)
    draw.rounded_rectangle(plot_box, radius=20, outline=mix_with_background(COLOR_PRIMARY, 0.22), width=2, fill=hex_to_rgb(COLOR_BACKGROUND))
    left, top, right, bottom = plot_box

    x = scatter["desvio_carga_pp"].astype(float)
    y = scatter["litros_mes"].astype(float)
    x_min = float(x.quantile(0.02) - 2)
    x_max = float(x.quantile(0.98) + 2)
    y_min = 0.0
    y_max = float(y.max() * 1.08)

    scatter_plot = scatter.loc[(x >= x_min - 1e-9) & (x <= x_max + 1e-9)].copy()

    def to_px(x_val: float, y_val: float) -> tuple[float, float]:
        px = left + (x_val - x_min) / (x_max - x_min) * (right - left)
        py = bottom - (y_val - y_min) / (y_max - y_min) * (bottom - top)
        return px, py

    y_ticks = np.linspace(y_min, y_max, 6)
    for y_tick in y_ticks:
        py = to_px(x_min, float(y_tick))[1]
        draw.line((left, py, right, py), fill=mix_with_background(COLOR_PRIMARY, 0.10), width=1)
        label = f"{y_tick/1_000_000:.1f} M" if y_tick >= 1_000_000 else f"{y_tick/1_000:.0f} mil"
        w, h = text_size(draw, label, tick_font)
        draw.text((left - 18 - w, py - h / 2), label, fill=hex_to_rgb(COLOR_PRIMARY), font=tick_font)

    x_ticks = np.linspace(x_min, x_max, 7)
    for x_tick in x_ticks:
        px = to_px(float(x_tick), y_min)[0]
        draw.line((px, top, px, bottom), fill=mix_with_background(COLOR_PRIMARY, 0.07), width=1)
        label = f"{x_tick:.0f}%"
        w, _ = text_size(draw, label, tick_font)
        draw.text((px - w / 2, bottom + 12), label, fill=hex_to_rgb(COLOR_PRIMARY), font=tick_font)

    zero_x = to_px(0.0, y_min)[0]
    draw.line((zero_x, top, zero_x, bottom), fill=mix_with_background(COLOR_PRIMARY, 0.25), width=2)
    draw.line((left, bottom, right, bottom), fill=mix_with_background(COLOR_PRIMARY, 0.35), width=2)
    draw.line((left, top, left, bottom), fill=mix_with_background(COLOR_PRIMARY, 0.35), width=2)

    slope, intercept = np.polyfit(scatter_plot["desvio_carga_pp"].to_numpy(), scatter_plot["litros_mes"].to_numpy(), 1)
    reg_start = to_px(x_min, intercept + slope * x_min)
    reg_end = to_px(x_max, intercept + slope * x_max)
    draw.line((reg_start[0], reg_start[1], reg_end[0], reg_end[1]), fill=hex_to_rgb(COLOR_ACCENT), width=4)

    for row in scatter_plot.itertuples(index=False):
        point_color = hex_to_rgb(COLOR_ACCENT if row.grupo_jurisdiccion == "CABA" else COLOR_PRIMARY)
        px, py = to_px(float(row.desvio_carga_pp), float(row.litros_mes))
        r = 8 if row.grupo_jurisdiccion == "CABA" else 6
        draw.ellipse((px - r, py - r, px + r, py + r), fill=point_color, outline=hex_to_rgb(COLOR_BACKGROUND))

    labels = choose_scatter_labels(scatter_plot)
    for row in labels.itertuples(index=False):
        px, py = to_px(float(row.desvio_carga_pp), float(row.litros_mes))
        dx = 12 if float(row.desvio_carga_pp) <= float(scatter_plot["desvio_carga_pp"].median()) else -12
        text = str(row.estacion).split(" (")[0]
        w, h = text_size(draw, text, label_font)
        tx = px + dx if dx > 0 else px + dx - w
        ty = py - h - 4
        draw.text((tx, ty), text, fill=hex_to_rgb(COLOR_PRIMARY), font=label_font, anchor=None)

    x_label = "Desvío de carga subnacional respecto del promedio CABA-Provincia (p.p.)"
    xw, _ = text_size(draw, x_label, axis_font)
    draw.text(((left + right - xw) / 2, 920), x_label, fill=hex_to_rgb(COLOR_PRIMARY), font=axis_font)
    draw.text((30, 470), "Litros por mes", fill=hex_to_rgb(COLOR_PRIMARY), font=axis_font)

    legend_x = 1400
    legend_y = 120
    draw.ellipse((legend_x, legend_y, legend_x + 16, legend_y + 16), fill=hex_to_rgb(COLOR_ACCENT))
    draw.text((legend_x + 24, legend_y - 2), "CABA", fill=hex_to_rgb(COLOR_PRIMARY), font=label_font)
    draw.ellipse((legend_x, legend_y + 30, legend_x + 16, legend_y + 46), fill=hex_to_rgb(COLOR_PRIMARY))
    draw.text((legend_x + 24, legend_y + 28), "Partido BA", fill=hex_to_rgb(COLOR_PRIMARY), font=label_font)

    corr = float(pd.Series(scatter_plot["desvio_carga_pp"]).corr(pd.Series(scatter_plot["litros_mes"])))
    corr_text = f"Correlación lineal: {corr:.2f}"
    box = (1280, 895, 1700, 950)
    draw.rounded_rectangle(box, radius=14, fill=mix_with_background(COLOR_PRIMARY, 0.03), outline=mix_with_background(COLOR_PRIMARY, 0.18), width=2)
    draw.text((1300, 910), corr_text, fill=hex_to_rgb(COLOR_PRIMARY), font=label_font)

    image.save(output_png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Renderiza mapa y scatter para la presentacion final.")
    parser.add_argument("--producto", default=DEFAULT_PRODUCT, help="Producto a analizar.")
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
    output_prefix = f"presentacion_{periodo.strftime('%Y-%m')}_{slugify(producto)}"

    estaciones = load_period_data(producto=producto, periodo=periodo)
    frontier_pairs = enrich_frontier_pairs(queries.compute_frontier_pairs(estaciones))
    map_pairs = aggregate_map_pairs(frontier_pairs)
    map_examples = select_map_examples(map_pairs)
    scatter = build_caba_prov_scatter_dataset(frontier_pairs)

    map_png = OUTPUT_DIR / f"{output_prefix}_mapa_frontera.png"
    scatter_png = OUTPUT_DIR / f"{output_prefix}_scatter_caba_vs_provincia.png"

    build_map_visual(estaciones=estaciones, map_examples=map_examples, output_png=map_png, periodo=periodo, producto=producto)
    build_scatter_visual(scatter=scatter, output_png=scatter_png, periodo=periodo, producto=producto)

    print("Visuales exportadas:")
    print(f"- {map_png}")
    print(f"- {scatter_png}")


if __name__ == "__main__":
    main()
