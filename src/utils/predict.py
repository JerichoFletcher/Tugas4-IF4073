import cv2
from segment_image import segment_image
from extract_features import extract_features
from visualize_bboxes import visualize_bboxes
import matplotlib.pyplot as plt

import joblib
model = joblib.load("svm_vehicle_detection_model.pkl")

def predict(image_path):
    class_names = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
    
    thresh_image, segmented_image = segment_image(image_path)
    features = extract_features(segmented_image)
    prediction = model.predict([features])[0]
    classname = class_names[prediction]

    original_image = cv2.imread(image_path)
    detected = visualize_bboxes(thresh_image, original_image, classname)
    return(detected)

image = predict("test.jpg")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
