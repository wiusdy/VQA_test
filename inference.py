from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers.utils import logging

class Inference:
    def __init__(self):
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.blip_model_saffal = BlipForQuestionAnswering.from_pretrained("wiusdy/blip_pretrained_saffal_fashion_finetuning")
        self.blip_model_control_net = BlipForQuestionAnswering.from_pretrained("wiusdy/blip_pretrained_control_net_fashion_finetuning")

        logging.set_verbosity_info()
        self.logger = logging.get_logger("transformers")

    def inference(self, image, text):
        self.logger.info(f"Running inference for model BLIP Saffal")
        BLIP_saffal_inference = self.__inference_saffal_blip(image, text)
        self.logger.info(f"Running inference for model BLIP Control Net")
        BLIP_control_net_inference = self.__inference_control_net_blip(image, text)
        return BLIP_saffal_inference, BLIP_control_net_inference

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