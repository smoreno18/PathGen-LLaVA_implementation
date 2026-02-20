import torch
from argparse import Namespace
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

# ---- configuración para PathGen-LLaVA ----
DEFAULT_ARGS = Namespace(
    conv_mode='llava_v1', # Crucial para evitar texto cortado
    max_new_tokens=512,
    image_aspect_ratio='pad', # Recomendado para patología
    model_base=None,
    model_path='/workspace/MLLMs/PathGen/PathGen-LLaVA', # Ruta local del contenedor
    temperature=0.2,
)

_MODEL_STATE = None

def load_llava_model(args: Namespace = None):
    global _MODEL_STATE
    if _MODEL_STATE is not None:
        return _MODEL_STATE

    if args is None:
        args = DEFAULT_ARGS

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    # Carga optimizada para PathGen
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name
    )

    _MODEL_STATE = (tokenizer, model, image_processor, context_len, args)
    return _MODEL_STATE