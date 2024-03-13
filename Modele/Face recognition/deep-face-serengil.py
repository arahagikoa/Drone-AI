from deepface import DeepFace
import matplotlib.pyplot as plt




img1_path = "./im1.jpg"
img2_path = "./im2.jpg"


img1 = DeepFace.detectFace(img1_path)
img2 = DeepFace.detectFace(img2_path)

#plt.imshow(img1)
#plt.imshow(img2)

model_name = 'VGG-Face'

resp = DeepFace.verify(img1_path, img2_path, model_name=model_name)

print(resp)


