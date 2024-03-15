from ultralytics import YOLO
import shutil
import os

model = YOLO("yolov8s.pt")
current_directory = os.path.dirname(os.path.abspath(__file__))

source_folder = os.path.join(current_directory, "photos_to_detect")
destination_folder = os.path.join(current_directory, "photos_with_detected_objects")
move_folder = os.path.join(current_directory, "runs", "detect", "predict")
remaining_folder = os.path.join(current_directory, "detected_without_frames")

os.makedirs(destination_folder, exist_ok=True)
os.makedirs(remaining_folder, exist_ok=True)

# Process images for detection
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
        
        # Move detected images
        detected_image = os.path.join(move_folder, filename)
        if os.path.exists(detected_image):
            shutil.move(detected_image, os.path.join(destination_folder, filename))

# Remove the detection folder
shutil.rmtree(move_folder)

# Move remaining images
for filename in os.listdir(source_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        source_image = os.path.join(source_folder, filename)
        shutil.move(source_image, os.path.join(remaining_folder, filename))
