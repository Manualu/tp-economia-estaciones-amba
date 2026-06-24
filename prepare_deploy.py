from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import AMBA_PROVINCES, CSV_DATASET_PATH, DEPLOY_DATASET_PATH


DEPLOY_COLUMNS = [
    "periodo",
    "operador",
    "nro_inscripcion",
    "bandera",
    "fecha_baja",
    "tipo_negocio",
    "direccion",
    "localidad",
    "provincia",
    "producto",
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
    "longitud",
    "latitud",
    "activa_en_periodo",
    "carga_subnacional_informada_pct",
    "tiene_desglose_impositivo",
    "iibb_referencia_texto",
    "iibb_referencia_min_pct",
    "iibb_referencia_max_pct",
    "tasa_vial_referencia_tipo",
    "tasa_vial_referencia_valor",
]


def main() -> None:
    if not CSV_DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontro el dataset base en {CSV_DATASET_PATH}")

    print(f"Leyendo dataset base: {CSV_DATASET_PATH}")
    df = pd.read_csv(CSV_DATASET_PATH, usecols=lambda column: column in DEPLOY_COLUMNS)

    filtered = df.loc[df["provincia"].isin(AMBA_PROVINCES)].copy()
    filtered = filtered[DEPLOY_COLUMNS]

    DEPLOY_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(DEPLOY_DATASET_PATH, index=False, compression={"method": "gzip", "compresslevel": 5})

    base_size_mb = CSV_DATASET_PATH.stat().st_size / (1024 * 1024)
    deploy_size_mb = DEPLOY_DATASET_PATH.stat().st_size / (1024 * 1024)
    reduction_pct = (1 - (deploy_size_mb / base_size_mb)) * 100 if base_size_mb else 0.0

    print("Dataset para deploy generado:")
    print(f"- Archivo: {DEPLOY_DATASET_PATH}")
    print(f"- Filas: {len(filtered):,}")
    print(f"- Tamano base: {base_size_mb:,.2f} MB")
    print(f"- Tamano deploy: {deploy_size_mb:,.2f} MB")
    print(f"- Reduccion: {reduction_pct:,.1f}%")


if __name__ == "__main__":
    main()
