from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the model
    # 'yolov8n.pt' is the "Nano" version (Smallest & Fastest)
    model = YOLO('yolov8n.pt') 

    # 2. Train the model
    print("Starting training...")
    results = model.train(
        data='data.yaml',   # Path to your config file
        epochs=20,          # How many times to loop over the data
        imgsz=640,          # Image size
        device='mps',       # Use 'mps' for Mac GPU acceleration. Change to 'cpu' if it fails.
        batch=8             # Lower batch size to save memory
    )
    print("Training finished!")
