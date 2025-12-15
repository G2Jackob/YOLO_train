from ultralytics import YOLO

model = YOLO("yolo11m.pt")

# Adjusted parameters for better training
model.train(
    data="C:\\Users\\papil\\YOLO\\YOLO\\data.yaml",
    epochs=100,        # Total training epochs
    imgsz=640,        # Image size
    batch=8,          # Batch size (reduce if running out of memory)
    workers=0,        # Number of worker threads
    device=0,         # GPU device (use "cpu" if no GPU)
    patience=50,      # Early stopping patience
    save=True,        # Save checkpoints
    verbose=True      # Print verbose output
)



