import os
import cv2
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

output_folder = './output'
os.makedirs(output_folder, exist_ok=True)
images_folder = './images'
json_path = './full_data.json' 
X = []
Y = []
label_mapping = {1: 'text', 2: 'figure', 3: 'table'}

def get_contour_features(image, bbox, label):
    x, y, w, h = map(int, bbox)
    y = max(0, min(y, image.shape[0] - 1))
    x = max(0, min(x, image.shape[1] - 1))
    h = min(h, image.shape[0] - y)
    w = min(w, image.shape[1] - x)
    
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:  
        return [0, 0, 0, 0, 0, 0, 0]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
        aspect_ratio = w_c / h_c if h_c != 0 else 0
        
        perimeter = cv2.arcLength(contour, True)
        compactness = (perimeter * perimeter) / (4 * np.pi * area) if area != 0 else 0
        
        if h_c < 4.2 and w_c > 190:  
            is_line = 1
        else:
            is_line = 0
        
        contour_features.append([
            area,
            aspect_ratio,
            w_c,
            h_c,
            perimeter,
            compactness,
            is_line
        ])
    
    if not contour_features:
        return [0, 0, 0, 0, 0, 0, 0]
    
    return np.mean(contour_features, axis=0)

def process_annotations(annotations_file):
    with open(annotations_file, 'r') as f:
        all_annotations = json.load(f)
    
    features = []
    labels = []
    
    for image_data in all_annotations:
        image_path = os.path.join(images_folder, image_data['file_name'])
        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_data['file_name']}")
            continue
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image - {image_data['file_name']}")
            continue
        
        for bbox, category_id in zip(image_data['bbox'], image_data['category_id']):
            if category_id not in label_mapping:
                continue
                
            bbox_features = get_contour_features(image, bbox, category_id)
            
            features.append(bbox_features)
            labels.append(category_id)
    
    return np.array(features), np.array(labels)

print("Starting annotation processing...")
X, Y = process_annotations(json_path)
print(f"Processed {len(X)} annotations in total")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Training model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, 'model.pkl')

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("\nModel Training Complete!")
print(f"Overall Accuracy: {accuracy:.2f}")
print(f"Overall Precision: {precision:.2f}")
print(f"Overall Recall: {recall:.2f}")
print(f"Overall F1-score: {f1:.2f}")

for category_id, label_name in label_mapping.items():
    mask = y_test == category_id
    if np.any(mask):  
        class_precision = precision_score(y_test == category_id, y_pred == category_id, average='binary')
        class_recall = recall_score(y_test == category_id, y_pred == category_id, average='binary')
        class_f1 = f1_score(y_test == category_id, y_pred == category_id, average='binary')
        
        print(f"\nMetrics for {label_name}:")
        print(f"Precision: {class_precision:.2f}")
        print(f"Recall: {class_recall:.2f}")
        print(f"F1-score: {class_f1:.2f}")
