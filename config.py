from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RAW_DATA_PATH = RAW_DIR / "Data.xlsx"
OVERRIDES_PATH = RAW_DIR / "overrides.csv"
IIBB_PATH = RAW_DIR / "IIBB_473003_por_provincia.xlsx"
TASA_VIAL_PATH = RAW_DIR / "Tasa_Vial_Buenos_Aires_2025.xlsx"

PARQUET_PATH = PROCESSED_DIR / "estaciones.parquet"
CSV_DATASET_PATH = PROCESSED_DIR / "estaciones.csv"
DEPLOY_DATASET_PATH = PROCESSED_DIR / "estaciones_streamlit.csv.gz"
CLEANING_REPORT_PATH = PROCESSED_DIR / "cleaning_report.csv"
PARTIDOS_AMBA_PATH = PROCESSED_DIR / "partidos_amba_ba.geojson"


def resolve_dataset_path() -> Path:
    for candidate in (DEPLOY_DATASET_PATH, CSV_DATASET_PATH):
        if candidate.exists():
            return candidate
    return DEPLOY_DATASET_PATH


DATASET_PATH = resolve_dataset_path()
AMBA_PROVINCES = [
    "Ciudad Autónoma de Buenos Aires",
    "Buenos Aires",
]

INITIAL_VIEW_STATE = {
    "latitude": -34.65,
    "longitude": -58.55,
    "zoom": 9,
    "pitch": 0,
    "bearing": 0,
}

ARGENTINA_BBOX = {
    "lat_min": -55.0,
    "lat_max": -21.0,
    "lon_min": -74.0,
    "lon_max": -53.0,
}

BANDERA_ALIASES = {
    "YPF": ["YPF", "YPF SA", "YPF S.A.", "YPF SOCIEDAD ANONIMA", "Ypf"],
    "Shell": ["Shell", "SHELL C.A.P.S.A.", "SHELL CAPSA", "Shell C.A.P.S.A."],
    "Axion": ["Axion", "AXION", "AXION ENERGY", "Axion Energy"],
    "Puma": ["Puma", "PUMA", "PUMA ENERGY", "Puma Energy"],
    "Refinor": ["Refinor", "REFINOR"],
    "DAPSA": ["DAPSA", "DAPSA S.A.", "DAPSA SA"],
    "Gulf": ["GULF", "Gulf"],
    "Voy": ["VOY", "Voy"],
    "Blanca": ["BLANCA", "Blanca", "SIN BANDERA", "INDEPENDIENTE", "-", "", None],
}

BANDERA_COLORS = {
    "YPF": [0, 59, 122, 220],
    "Shell": [255, 203, 5, 220],
    "Axion": [227, 6, 19, 220],
    "Puma": [27, 58, 109, 220],
    "Refinor": [245, 130, 32, 220],
    "DAPSA": [0, 107, 63, 220],
    "Gulf": [255, 111, 0, 220],
    "Voy": [0, 150, 136, 220],
    "Blanca": [158, 158, 158, 200],
    "Otra": [106, 27, 154, 220],
}

PRODUCT_ALIASES = {
    "nafta (super) entre 92 y 95 ron": "Nafta Súper",
    "nafta (premium) de mas de 95 ron": "Nafta Premium",
    "nafta (comun) hasta 92 ron": "Nafta Común",
    "gas oil grado 2": "Gas Oil Grado 2",
    "gas oil grado 3": "Gas Oil Grado 3",
    "gnc": "GNC",
    "kerosene": "Kerosene",
    "glpa": "GLPA",
}

PRODUCT_TO_TASA_VIAL_COLUMN = {
    "Gas Oil Grado 2": "diesel_g2",
    "Gas Oil Grado 3": "diesel_g3",
    "Nafta Súper": "nafta_g2",
    "Nafta Premium": "nafta_g3",
    "Nafta Común": "otros",
    "Kerosene": "otros",
    "GLPA": "otros",
    "GNC": "gnc",
}

PROVINCE_ALIASES = {
    "BUENOS AIRES": "Buenos Aires",
    "CAPITAL FEDERAL": "Ciudad Autónoma de Buenos Aires",
    "CIUDAD AUTONOMA DE BUENOS AIRES": "Ciudad Autónoma de Buenos Aires",
    "CABA": "Ciudad Autónoma de Buenos Aires",
}

DEFAULT_PRODUCT = "Nafta Súper"
FRONTIER_DISTANCE_KM = 2.0
