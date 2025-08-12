#To check the class ID and class name for trained models

from ultralytics import YOLO

model_path = r"C:\thowl\AUV\Carla\better.pt"
model = YOLO(model_path)

# Print class IDs and names
print("Class IDs and Names:")
for class_id, class_name in model.model.names.items():
    print(f"ID: {class_id}, Name: {class_name}")
