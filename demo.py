from main import pathgen_infer_region
from utils.load_model import load_llava_model
import PIL.Image as Image

# Cargar configuraciÃ³n y modelo
tokenizer, model, image_processor, context_len, args = load_llava_model()

# Preparar imagen de prueba
wsi_path = "/workspace/BBDD/HCUV_0017_region_1.jpg"
slide = Image.open(wsi_path).convert("RGB")

x1, y1 = 1027, 1949
coords = (x1, x1+512, y1, y1+512)
prompt = "Which patterns can you observe in this skin tissue?"

# Inferencia
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