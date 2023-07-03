import cv2
from PIL import Image
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()


def detect_people(image):
    img = Image.fromarray(image)
    img = img.convert('RGB')
    img = img.resize((640, 640))
    results = model(img)
    detections = results.xyxy[0]
    detections = detections[detections[:, 5] == 0]  # Filter only people (class index 0)
    return detections


average_pixel_height = 150


def calculate_size(pixel_size, focal_length):
    return (pixel_size * focal_length) / average_pixel_height


image = cv2.imread('100.jpg')

detections = detect_people(image)

for detection in detections:
    xmin, ymin, xmax, ymax, _, confidence = detection[:6]
    pixel_height = ymax - ymin
    focal_length = 100  # Replace with your focal length in millimeters
    centimeter_height = calculate_size(pixel_height, focal_length)

    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    text_width, text_height = cv2.getTextSize(
        f"Height: {centimeter_height:.2f} cm", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    if ymin - 10 - text_height > 0:
        text_y = ymin - 10
    else:
        text_y = ymin + 10 + text_height

    cv2.putText(image, f"Height: {centimeter_height:.2f} cm", (int(xmin), int(text_y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    print("===============")
    print(f"Height: {centimeter_height:.2f} cm")

cv2.imwrite("output.jpg", image)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
