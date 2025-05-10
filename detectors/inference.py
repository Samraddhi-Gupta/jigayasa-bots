import cv2
import numpy
from PIL import Image as image_main
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import matplotlib.pyplot as plt

image_path = './processed_image.jpg'
model_path = 'model_final.pth'
model_zoo_config_name = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
prediction_score_threshold = 0.7
class_labels = ['text', 'title', 'list', 'table', 'figure']

colors = {
    'text': (0, 1, 0),     
    'title': (1, 0, 0),    
    'list': (1, 1, 0),     
    'table': (1, 0, 1),    
    'figure': (0, 0, 1)    
}

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

predictor = DefaultPredictor(cfg)

image_pil = image_main.open(image_path)
image_cv = cv2.cvtColor(numpy.array(image_pil), cv2.COLOR_RGB2BGR)
outputs = predictor(image_cv)

visualization_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))
plt.imshow(visualization_image)
current_axis = plt.gca()

instances = outputs["instances"].to("cpu")
pred_boxes = instances.pred_boxes
scores = instances.scores
pred_classes = instances.pred_classes

for i in range(0, len(pred_boxes)):
    box = pred_boxes[i].tensor.numpy()[0]
    score = round(float(scores[i].numpy()), 4)
    label_key = int(pred_classes[i].numpy())
    label = class_labels[label_key]
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    w = x2 - x1
    h = y2 - y1
    color = colors[label]
    rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=2)
    current_axis.add_patch(rect)
    
    plt.text(x1, y1-5, f'{label}: {score:.2f}', 
             bbox=dict(facecolor=color, alpha=0.7),
             color='white', fontsize=8)
    
    print(f'Detected {label} with score={score:.4f} at box=[{x1}, {y1}, {w}, {h}]')

plt.axis('off')

plt.tight_layout()

plt.savefig('output_visualization.png', bbox_inches='tight', dpi=300)

plt.show()
