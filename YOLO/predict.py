from ultralytics import YOLO
import os
import random

# Get path to test images folder
test_images_path = "C:\\Users\\papil\\YOLO\\YOLO\\test\\images"

# Get list of all images in the folder
image_files = [f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Randomly select an image
random_image = random.choice(image_files)

# Load model and make prediction
model = YOLO("YOLO/tree_detect_yolo.pt")
results = model.predict(source=os.path.join(test_images_path, random_image), show=True, save=True, line_width=1, conf=0.6)