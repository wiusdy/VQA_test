from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering
from transformers.utils import logging

import torch

class Inference:
    def __init__(self):
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.blip_model = BlipForQuestionAnswering.from_pretrained("wiusdy/blip_pretrained_saffal_fashion_finetuning").to("cuda")
        logging.set_verbosity_info()
        self.logger = logging.get_logger("transformers")

    def inference(self, selected, image, text):
        self.logger.info(f"selected model {selected}")
        if selected == "Model 1":
            return self.__inference_vilt(image, text)
        elif selected == "Model 2":
            return self.__inference_deplot(image, text)
        else:
            self.logger.warning("Please select a model to make the inference..")

    def __inference_vilt(self, image, text):
        encoding = self.vilt_processor(image, text, return_tensors="pt")
        outputs = self.vilt_model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return f"{self.vilt_model.config.id2label[idx]}"

    def __inference_deplot(self, image, text):
        encoding = self.blip_processor(image, text, return_tensors="pt").to("cuda:0", torch.float16)
        out = self.blip_model.generate(**encoding)
        generated_text = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return f"{generated_text}"