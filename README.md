# PathGen-LLaVA – Inferencia sobre region de WSI
Este repo contiene una versión de **PathGen-LLaVA** adaptada para hacer
inferencia sobre regiones de Whole Slide Images (WSI), scripts claves:

- `utils/load_model.py` → carga del modelo
- `main.py`      → función de inferencia sobre una región de WSI
- `demo.py`              → **demo** de uso 


## Construir imagen docker y lanzar contenedor

```bash
# Build
docker build -t quilt-llava-qllava .

# Run 
docker run <CONFIG> quilt-llava-qllava
```

## Descargar los pesos desde HF

```bash
# Descarga en carpeta local
huggingface-cli download jamessyx/PathGen-LLaVA --local-dir <llava_dir>/PathGen-LLaVA
```