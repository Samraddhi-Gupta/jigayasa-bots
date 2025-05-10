import cv2
import numpy as np
import joblib

def get_contour_features(contour):
    area = cv2.contourArea(contour)
    x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
    aspect_ratio = w_c / h_c if h_c != 0 else 0
    perimeter = cv2.arcLength(contour, True)
    compactness = (perimeter * perimeter) / (4 * np.pi * area) if area != 0 else 0
    is_line = 1 if (h_c < 4.2 and w_c > 190) else 0
    
    return [area, aspect_ratio, w_c, h_c, perimeter, compactness, is_line]

def group_lines_by_length(lines, line_grouping_threshold=3):
    grouped_lines = []
    if not lines:
        return grouped_lines
        
    lines.sort(key=lambda x: (x[2], x[0])) 
    current_group = [lines[0]]
    
    for i in range(1, len(lines)):
        if (abs(lines[i][2] - current_group[0][2]) <= line_grouping_threshold and
            abs(lines[i][0] - current_group[0][0]) <= line_grouping_threshold):
            current_group.append(lines[i])
        else:
            grouped_lines.append(current_group)
            current_group = [lines[i]]
    
    if current_group:
        grouped_lines.append(current_group)
    
    return grouped_lines

def draw_label(image, label, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = x
    text_y = y - 10
    cv2.rectangle(image, (text_x, text_y - text_size[1]), 
                 (text_x + text_size[0], text_y + 5), (0, 0, 0), -1)
    cv2.putText(image, label, (text_x, text_y), font, 
                font_scale, (255, 255, 255), font_thickness)

def analyze_document_layout(image_path, model_path, output_path):
    clf = joblib.load(model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    grouped_contours = []  
    horizontal_lines = []  
    label_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0)}
    label_names = {1: 'Text', 2: 'Figure', 3: 'Table'}
    
    for contour in contours:
        features = get_contour_features(contour)
        prediction = clf.predict([features])[0]
        x, y, w, h = cv2.boundingRect(contour)
        
        if prediction == 1:  # Text
            merged = False
            for i, (prev_x, prev_y, prev_w, prev_h) in enumerate(grouped_contours):
                if abs(prev_y - y) <= 15:  
                    new_x = min(x, prev_x)
                    new_y = min(y, prev_y)
                    new_w = max(x + w, prev_x + prev_w) - new_x
                    new_h = max(y + h, prev_y + prev_h) - new_y
                    grouped_contours[i] = (new_x, new_y, new_w, new_h)
                    merged = True
                    break
            
            if not merged:
                grouped_contours.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), label_colors[prediction], 2)
            
        elif prediction == 2:  # Figure
            cv2.rectangle(image, (x, y), (x+w, y+h), label_colors[prediction], 2)
            
        else:  # Table
            horizontal_lines.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x+w, y+h), label_colors[prediction], 2)
    
    for (x, y, w, h) in grouped_contours:
        cv2.rectangle(image, (x, y), (x + w, y + h), label_colors[1], 2)
        draw_label(image, label_names[1], x, y, label_colors[1])
    
    for (x, y, w, h) in grouped_contours:
        draw_label(image, label_names[1], x, y, label_colors[1])
    
    grouped_lines = group_lines_by_length(horizontal_lines)
    for group in grouped_lines:
        if len(group) >= 2:  
            x_coords = [line[0] for line in group]
            y_coords = [line[1] for line in group]
            
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x + w for x, _, w, _ in group)
            y_max = max(y + h for _, y, _, h in group)
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), label_colors[3], 2)
            draw_label(image, label_names[3], x_min, y_min, label_colors[3])

    cv2.imwrite(output_path, image)
    print(f"Analysis complete. Output saved to: {output_path}")
    return image

model_path = 'model.pkl'
image_path = 'adr-croped.png'  
output_path = f'./output/analyzed_{image_path}'

try:
    analyzed_image = analyze_document_layout(image_path, model_path, output_path)
    print("Document analysis completed successfully")
except Exception as e:
    print(f"Error during analysis: {str(e)}")
