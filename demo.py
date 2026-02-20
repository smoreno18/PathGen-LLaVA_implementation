from main import pathgen_infer_region
# Asumiendo que tienes una utilidad similar para cargar el modelo
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import PIL.Image as Image
import argparse

# Configuración de argumentos simulada para el modelo
class Args:
    def __init__(self):
        self.conv_mode = "llava_v1"
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.image_aspect_ratio = 'pad' # Importante para LLaVA v1.5

# 1. Cargar el modelo PathGen-LLaVA
model_path = "/workspace/MLLMs/PathGen-LLaVA"
model_name = get_model_name_from_path(model_path)

# Cargamos el modelo usando la función estándar de LLaVA
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device_map="auto"
)

args = Args()

# 2. Preparar la WSI
wsi_path = "/workspace/BBDD/HCUV_0017_region_1.jpg"
slide = Image.open(wsi_path)

# Definir coordenadas (x1, x2, y1, y2)
x1, y1 = 1027, 1949
coords = (x1, x1+512, y1, y1+512)

prompt = "Which patterns do you observe in this skin tissue?"

# 3. Inferencia
resp, conv = pathgen_infer_region(
    tokenizer, model, image_processor, context_len, args,
    wsi=slide,
    coords=coords,
    user_prompt=prompt,
    level=0,
    conv=None,
)

print(f"\n[USER]:\n{prompt}")
print(f"\n[ASSISTANT]:\n{resp}")