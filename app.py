import gradio as gr
import os

from inference import Inference

inference = Inference()


with gr.Blocks() as block:
    txt = gr.Textbox(label="Insert a question..", lines=2)
    outputs = [gr.outputs.Textbox(label="Answer from BLIP saffal model"), gr.outputs.Textbox(label="Answer from BLIP control net")]

    btn = gr.Button(value="Submit")

    dogs = os.path.join(os.path.dirname(__file__), "1.png")
    image = gr.Image(type="pil", value=dogs)

    btn.click(inference.inference, inputs=[image, txt], outputs=outputs)

if __name__ == "__main__":
    block.launch()