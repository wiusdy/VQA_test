from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering
from transformers.utils import logging

import torch

class Inference:
    def __init__(self):
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vilt_model_saffal = BlipForQuestionAnswering.from_pretrained("wiusdy/vilt_saffal_model")
        self.vilt_model_control_net = BlipForQuestionAnswering.from_pretrained("wiusdy/vilt_control_net")

        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.blip_model_saffal = BlipForQuestionAnswering.from_pretrained("wiusdy/blip_pretrained_saffal_fashion_finetuning")
        self.blip_model_control_net = BlipForQuestionAnswering.from_pretrained("wiusdy/blip_pretrained_control_net_fashion_finetuning")

        logging.set_verbosity_info()
        self.logger = logging.get_logger("transformers")

    def inference(self, image, text):
        self.logger.info(f"Running inference for model ViLT Saffal")
        ViLT_saffal_inference = self.__inference_vilt_saffal(image, text)
        self.logger.info(f"Running inference for model ViLT Control Net")
        ViLT_control_net_inference = self.__inference_vilt_control_net(image, text)
        self.logger.info(f"Running inference for model BLIP Saffal")
        BLIP_saffal_inference = self.__inference_saffal_blip(image, text)
        self.logger.info(f"Running inference for model BLIP Control Net")
        BLIP_control_net_inference = self.__inference_control_net_blip(image, text)
        return BLIP_saffal_inference, BLIP_control_net_inference, ViLT_saffal_inference, ViLT_control_net_inference

    def __inference_vilt_saffal(self, image, text):
        encoding = self.vilt_processor(image, text, return_tensors="pt")
        out = self.vilt_model_saffal.generate(**encoding)
        generated_text = self.vilt_processor.decode(out[0], skip_special_tokens=True)
        return f"{generated_text}"

    def __inference_vilt_control_net(self, image, text):
        encoding = self.vilt_processor(image, text, return_tensors="pt")
        out = self.vilt_model_control_net.generate(**encoding)
        generated_text = self.vilt_processor.decode(out[0], skip_special_tokens=True)
        return f"{generated_text}"

    def __inference_saffal_blip(self, image, text):
        encoding = self.blip_processor(image, text, return_tensors="pt")
        out = self.blip_model_saffal.generate(**encoding)
        generated_text = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return f"{generated_text}"

    def __inference_control_net_blip(self, image, text):
        encoding = self.blip_processor(image, text, return_tensors="pt")
        out = self.blip_model_control_net.generate(**encoding)
        generated_text = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return f"{generated_text}"