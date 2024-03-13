from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import math


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('facebook/dino-vits8')
model = ViTModel.from_pretrained('facebook/dino-vits8')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
features = last_hidden_states.detach().numpy()
print(features.shape)
