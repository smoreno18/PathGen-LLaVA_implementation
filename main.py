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

def pathgen_infer_region(
    tokenizer,
    model,
    image_processor,
    context_len,
    args,
    wsi: WSIType,
    coords: Tuple[int, int, int, int],
    user_prompt: str,
    level: int = 0,
    conv=None,
):
    """
    Inferencia adaptada específicamente para PathGen-LLaVA.
    """
    x1, y1, x2, y2 = coords
    patch = crop_region_from_wsi(wsi, x1, y1, x2, y2, level=level)

    # PathGen utiliza float16 o bfloat16 dependiendo del hardware
    image_tensor = process_images([patch], image_processor, args)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # PathGen-LLaVA usa típicamente el template 'llava_v1' o 'v1'
    if conv is None:
        conv_mode = getattr(args, "conv_mode", "llava_v1")
        conv = conv_templates[conv_mode].copy()

    roles = conv.roles
    inp = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
    conv.append_message(roles[0], inp)
    conv.append_message(roles[1], None)

    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=getattr(args, "temperature", 0.2),
            max_new_tokens=getattr(args, "max_new_tokens", 512),
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    generated_ids = output_ids[0, input_ids.shape[1]:]
    # En lugar de recortar manualmente, deja que el tokenizer gestione los tokens especiales
    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # Si la respuesta incluye tu pregunta al principio, límpiala:
    if prompt in outputs:
        outputs = outputs.replace(prompt, "").strip()
    conv.messages[-1][-1] = outputs

    return outputs, conv