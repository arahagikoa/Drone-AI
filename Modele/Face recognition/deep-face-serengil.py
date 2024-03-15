
import matplotlib.pyplot as plt




from deepface import DeepFace

img1_path = "./im1.jpg"
img2_path = "./im2.jpg"

model_name = 'VGG-Face'

resp = DeepFace.verify(img1_path, img2_path, model_name=model_name)

print(resp)
