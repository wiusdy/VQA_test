# This is a simple VQA system using Hugging Face, PyTorch and VQA models
-------------

In this repository we created a simple VQA system capable of recognize spatial and context information of fashion images (e.g. clothes color and details). 

The project was based in this paper **FashionVQA: A Domain-Specific Visual Question Answering System** [[1]](#1). We also used the VQA pre-trained model from **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** [[2]](#2) to make the model finetuning the two new models. 

We used the datasets **Deep Fashion with Masks** available in <https://huggingface.co/datasets/SaffalPoosh/deepFashion-with-masks> and the **Control Net Dataset** available in <https://huggingface.co/datasets/ldhnam/deepfashion_controlnet>.


## References
<a id="1">[1]</a> 
Min Wang and Ata Mahjoubfar and Anupama Joshi, 2022
FashionVQA: A Domain-Specific Visual Question Answering System

<a id="2">[2]</a> 
Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi, 2022
BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation