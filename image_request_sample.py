from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# prepare image + question
image_path = "617.jpg"
image = Image.open(image_path)
text = "What are the dogs riding?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
encoding = processor(image, text, return_tensors="pt")

outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print(f"{text}:", model.config.id2label[idx])