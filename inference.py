from transformers import AutoProcessor, BlipForQuestionAnswering
from transformers.utils import logging

class Inference:
    def __init__(self):
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.blip_model_saffal = BlipForQuestionAnswering.from_pretrained("wiusdy/blip_pretrained_saffal_fashion_finetuning")
        self.blip_model_control_net = BlipForQuestionAnswering.from_pretrained("wiusdy/blip_pretrained_control_net_fashion_finetuning")
        logging.set_verbosity_info()
        self.logger = logging.get_logger("transformers")

    def inference(self, selected, image, text):
        self.logger.info(f"selected model {selected}, image shape {image.type}, question {text.value}")
        if selected == "Blip Saffal":
            return self.__inference_saffal_blip(image, text)
        elif selected == "Blip CN":
            return self.__inference_control_net_blip(image, text)
        else:
            self.logger.warning("Please select a model to make the inference..")

    def __inference_saffal_blip(self, image, text):
        encoding = self.blip_processor(image, text, return_tensors="pt")
        out = self.blip_model_saffal.generate(**encoding, max_new_tokens=100)
        generated_text = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return f"{generated_text}"

    def __inference_control_net_blip(self, image, text):
        encoding = self.blip_processor(image, text, return_tensors="pt")
        out = self.blip_model_control_net.generate(**encoding, max_new_tokens=100)
        generated_text = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return f"{generated_text}"