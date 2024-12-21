from utils.segment_image import segment_image
from utils.extract_features import extract_features
from utils.visualize_bboxes import visualize_bboxes

import joblib
model = joblib.load("models/svm_vehicle_detection_model.pkl")

def predict(img):
    class_names = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
    
    thresh_image, segmented_image = segment_image(img)
    features = extract_features(segmented_image)
    prediction = model.predict([features])[0]
    classname = class_names[prediction]

    detected = visualize_bboxes(thresh_image, img, classname)
    return(detected)
