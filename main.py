from typing import Tuple, Union
import torch
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)

try:
    import openslide
except ImportError:
    openslide = None

WSIType = Union["openslide.OpenSlide", Image.Image]

def crop_region_from_wsi(
    wsi: WSIType,
    x1: int, y1: int, x2: int, y2: int,
    level: int = 0,
) -> Image.Image:
    w, h = x2 - x1, y2 - y1
    if isinstance(wsi, Image.Image):
        return wsi.crop((x1, y1, x2, y2)).convert("RGB")
    if openslide is not None and isinstance(wsi, openslide.OpenSlide):
        return wsi.read_region((x1, y1), level, (w, h)).convert("RGB")
    raise TypeError(f"Objeto WSI no soportado: {type(wsi)}")

def pathgen_infer_region(tokenizer, model, image_processor, context_len, args, wsi, coords, user_prompt, level=0, conv=None):
    x1, y1, x2, y2 = coords
    # Extraer patch
    patch = wsi.crop((x1, y1, x2, y2)).convert("RGB") if hasattr(wsi, 'crop') else wsi.read_region((x1, y1), level, (x2-x1, y2-y1)).convert("RGB")

    image_tensor = process_images([patch], image_processor, args)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    if conv is None:
        conv = conv_templates[args.conv_mode].copy()

    inp = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    # En lugar de recortar manualmente, deja que el tokenizer gestione los tokens especiales
    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # Si la respuesta incluye tu pregunta al principio, límpiala:
    if prompt in outputs:
        outputs = outputs.replace(prompt, "").strip()
    conv.messages[-1][-1] = outputs

    return outputs, conv