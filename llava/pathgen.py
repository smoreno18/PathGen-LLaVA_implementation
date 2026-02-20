import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
from io import BytesIO

class PathGenLlavaPredictor:
    def __init__(self, model_path="pathfoundation/pathgen-llava-v1.5-7b"):
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)
        
        # Carga del modelo, tokenizer y procesador de imagen
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.conv_mode = "llava_v1" # PathGen suele basarse en v1.5

    def predict(self, image_path: str, prompt: str):
        # 1. Preparar el Prompt siguiendo el formato LLaVA
        conv = conv_templates[self.conv_mode].copy()
        roles = conv.roles
        
        # Insertar el token de imagen en el mensaje
        content = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(roles[0], content)
        conv.append_message(roles[1], None)
        prompt_final = conv.get_prompt()

        # 2. Cargar y procesar la imagen (Pathology Patch)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        # 3. Tokenización
        input_ids = tokenizer_image_token(prompt_final, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 4. Generación
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        # 5. Decodificación
        output = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
        return output

# Ejemplo de uso
if __name__ == "__main__":
    predictor = PathGenLlavaPredictor(model_path="/workspace/MLLMs/PathGen-LLaVA")
    res = predictor.predict("/workspace/Projects/AI4SKIN/test.png", "Which patterns do you observe in this skin tissue?")
    print(res)