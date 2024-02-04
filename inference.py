from transformers import ViltProcessor, ViltForQuestionAnswering, Pix2StructProcessor, Pix2StructForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from transformers.utils import logging

class Inference:
    def __init__(self):
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        self.deplot_processor = Pix2StructProcessor.from_pretrained('google/deplot')
        self.deplot_model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
        logging.set_verbosity_info()
        self.logger = logging.get_logger("transformers")

    def inference(self, selected, image, text):
        self.logger.info(f"selected model {selected}")
        if selected == "Model 1":
            return self.__inference_vilt(image, text)
        elif selected == "Model 2":
            return self.__inference_deplot(image, text)
        elif selected == "Model 3":
            return self.__inference_vilt(image, text)
        else:
            self.logger.warning("Please select a model to make the inference..")

    def __inference_vilt(self, image, text):
        encoding = self.vilt_processor(image, text, return_tensors="pt")
        outputs = self.vilt_model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return f"{self.vilt_model.config.id2label[idx]}"

    def __inference_deplot(self, image, text):
        inputs = self.deplot_processor(images=image, text=text, return_tensors="pt")
        predictions = self.deplot_model.generate(**inputs, max_new_tokens=512)
        return f"{self.deplot_processor.decode(predictions[0], skip_special_tokens=True)}"