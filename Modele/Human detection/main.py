from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




# First image
url1 = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image1 = Image.open(requests.get(url1, stream=True).raw)

# Second image
url2 = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image2 = Image.open(requests.get(url2, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Process the first image
inputs1 = processor(images=image1, return_tensors="pt")
outputs1 = model(**inputs1)
last_hidden_states1 = outputs1.last_hidden_state
features1 = last_hidden_states1.detach().numpy().squeeze(0)

# Process the second image
inputs2 = processor(images=image2, return_tensors="pt")
outputs2 = model(**inputs2)
last_hidden_states2 = outputs2.last_hidden_state
features2 = last_hidden_states2.detach().numpy().squeeze(0)

print("Shape of features1:", features1.shape)
print("Shape of features2:", features2.shape)

# Set up subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first image
axs[0, 0].imshow(image1)
axs[0, 0].set_title('Image 1')
axs[0, 0].axis('off')

# Plot the patches for the first image
overlay_image1 = image1.copy()
patch_size1 = int(np.sqrt(features1.shape[0]))
patch_height1 = features1.shape[1]
patch_width1 = 1  # Since there is no width dimension in features1
for i in range(patch_size1):
    for j in range(patch_size1):
        patch = features1[i * patch_size1 + j].reshape(patch_height1, patch_width1)
        patch_image = Image.fromarray((patch * 255).astype(np.uint8), 'L')
        overlay_image1.paste(patch_image, (j * patch_width1, i * patch_height1))
axs[0, 1].imshow(overlay_image1)
axs[0, 1].set_title('Patches for Image 1')
axs[0, 1].axis('off')

# Plot the second image
axs[1, 0].imshow(image2)
axs[1, 0].set_title('Image 2')
axs[1, 0].axis('off')

# Plot the patches for the second image
overlay_image2 = image2.copy()
patch_size2 = int(np.sqrt(features2.shape[0]))
patch_height2 = features2.shape[1]
patch_width2 = 1  # Since there is no width dimension in features2
for i in range(patch_size2):
    for j in range(patch_size2):
        patch = features2[i * patch_size2 + j].reshape(patch_height2, patch_width2)
        patch_image = Image.fromarray((patch * 255).astype(np.uint8), 'L')
        overlay_image2.paste(patch_image, (j * patch_width2, i * patch_height2))
axs[1, 1].imshow(overlay_image2)
axs[1, 1].set_title('Patches for Image 2')
axs[1, 1].axis('off')

# Show the plots
plt.show()


features1_2d = features1.reshape(features1.shape[0], -1)
features2_2d = features2.reshape(features2.shape[0], -1)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(features1_2d, features2_2d)

# Print the cosine similarity
print("Similarity:", similarity_matrix[0, 0])