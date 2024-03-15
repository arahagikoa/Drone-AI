from ultralytics import YOLO
import shutil
import os

model = YOLO("yolov8s.pt")

source_folder = "Modele\\Human detection\\photos_to_detect"
destination_folder = "Modele\\Human detection\\photos_with_detected_objects"
move_folder = "runs\\detect\\predict"
remaining_folder = "Modele\\Human detection\\detected_without_frames"

os.makedirs(destination_folder, exist_ok=True)
os.makedirs(remaining_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        source_image = os.path.join(source_folder, filename)
        
        model.predict(
            source=source_image, 
            show=True,  
            save=True,  
            save_dir=destination_folder,
            save_txt=False, 
            save_crop=False,  
            box=True,  
            visualize=False 
        )
        
        # Move the detected image
        detected_image = os.path.join(move_folder, filename)
        if os.path.exists(detected_image):
            shutil.move(detected_image, os.path.join(destination_folder, filename))

# Remove the detection folder
shutil.rmtree(move_folder)

# Move remaining images from photos_to_detect to detected_without_frames
for filename in os.listdir(source_folder):
    source_image = os.path.join(source_folder, filename)
    shutil.move(source_image, os.path.join(remaining_folder, filename))
