import gradio as gr
import os

from inference import Inference

inference = Inference()


with gr.Blocks() as block:
    options = gr.Dropdown(choices=["ViLT", "Blip Saffal", "Blip CN"], label="Models", info="Select the model to use..", )
    # need to improve this one...

    txt = gr.Textbox(label="Insert a question..", lines=2)
    txt_3 = gr.Textbox(value="", label="Your answer is here..")
    btn = gr.Button(value="Submit")

    dogs = os.path.join(os.path.dirname(__file__), "617.jpg")
    image = gr.Image(type="pil", value=dogs)

    btn.click(inference.inference, inputs=[options, image, txt], outputs=[txt_3])

if __name__ == "__main__":
    block.launch()