from ultralytics import YOLO
import cv2

# Load your model
model = YOLO('runs/detect/train/weights/best.pt') 

# USE A LOCAL FILE instead of a URL
# Make sure test_trash.jpg is in the same folder as this script
results = model.predict(source='test_trash.jpg', save=True, conf=0.25)

# Show the results
for result in results:
    img_with_boxes = result.plot()
    cv2.imshow('Wander Bin AI Test', img_with_boxes)
    print("Success! Check the popup window or the 'runs/detect/predict' folder.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
