# Estaciones de servicio en AMBA

App en Streamlit para explorar precio, volumen y desglose impositivo disponible de estaciones de servicio minoristas, con foco en CABA y Provincia de Buenos Aires.

## Archivos esperados

- `data/raw/Data.xlsx`: dataset principal.
- `data/raw/IIBB_473003_por_provincia.xlsx`: referencia provincial de IIBB.
- `data/raw/Tasa_Vial_Buenos_Aires_2025.xlsx`: referencia municipal de tasa vial.
- `data/raw/overrides.csv`: correcciones manuales y drops trazables.

## Cómo correr

1. Ir a la carpeta del proyecto, crear y activar entorno virtual:

```bash
cd "C:\Users\manua\OneDrive\Documentos\TP Economía"
python -m venv .venv
.venv\Scripts\activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Generar el dataset limpio:

```bash
python preprocess.py
```

4. Levantar la app:

```bash
streamlit run app.py
```

## Qué hace el preprocess

- Renombra columnas y normaliza formatos.
- Corrige coordenadas usando `geojson` cuando existe y escalado automático cuando no.
- Aplica limpieza obligatoria:
  - hard rules.
  - logic rules.
- Para esta versión de demo, la limpieza IQR avanzada quedó desactivada para priorizar velocidad de procesamiento sobre el Excel grande.
- Aplica `overrides.csv` para updates y drops auditables.
- Genera:
  - `data/processed/estaciones.csv`
  - `data/processed/cleaning_report.csv`

## Nota metodológica

La vista de hipótesis de frontera es un proxy exploratorio. Como el dataset actual no trae un campo de municipio uniforme ni polígonos jurisdiccionales, la app compara estaciones cercanas con distinta carga subnacional informada para aproximar el posible efecto de frontera.
