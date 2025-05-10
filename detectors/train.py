import os
import json
import cv2
from sklearn.model_selection import train_test_split
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode

def validate_and_remap_categories(annotation_file):
    with open(annotation_file, 'r') as f:
        dataset_dict = json.load(f)
    used_category_ids = set(anno['category_id'] for anno in dataset_dict['annotations'])
    
    sorted_category_ids = sorted(used_category_ids)
    
    category_remap = {old_id: new_id for new_id, old_id in enumerate(sorted_category_ids)}
    
    for anno in dataset_dict['annotations']:
        anno['category_id'] = category_remap[anno['category_id']]
    
    category_map = {cat['id']: cat for cat in dataset_dict['categories']}
    updated_categories = []
    for new_id, old_id in enumerate(sorted_category_ids):
        cat = category_map[old_id]
        cat['id'] = new_id
        updated_categories.append(cat)
    
    dataset_dict['categories'] = updated_categories
    
    return dataset_dict, [cat['name'] for cat in updated_categories], category_remap

def get_dataset_dicts(img_dir, annotation_file, is_train=True, train_split=0.8):
    dataset_dict, class_names, _ = validate_and_remap_categories(annotation_file)
    image_annotations = {}
    for anno in dataset_dict['annotations']:
        if anno['image_id'] not in image_annotations:
            image_annotations[anno['image_id']] = []
        image_annotations[anno['image_id']].append(anno)

    all_entries = []
    for img_info in dataset_dict['images']:
        record = {
            'file_name': os.path.join(img_dir, img_info['file_name']),
            'image_id': img_info['id'],
            'height': img_info['height'],
            'width': img_info['width']
        }
        
        annotations = image_annotations.get(img_info['id'], [])
        
        objs = []
        for anno in annotations:
            obj = {
                'bbox': anno['bbox'],
                'bbox_mode': BoxMode.XYWH_ABS,
                'category_id': anno['category_id'],
                'segmentation': anno.get('segmentation', [])
            }
            objs.append(obj)
        
        record['annotations'] = objs
        all_entries.append(record)
    
    train_entries, test_entries = train_test_split(all_entries, train_size=train_split, random_state=42)
    
    return train_entries if is_train else test_entries

def register_dataset(img_dir, annotation_file):
    dataset_dict, class_names, _ = validate_and_remap_categories(annotation_file)
    num_classes = len(class_names)
    dataset_train_name = 'custom_train'
    DatasetCatalog.register(dataset_train_name, 
                            lambda: get_dataset_dicts(img_dir, annotation_file, is_train=True))
    MetadataCatalog.get(dataset_train_name).set(
        thing_classes=class_names,
        evaluator_type='coco'
    )
    dataset_test_name = 'custom_test'
    DatasetCatalog.register(dataset_test_name, 
                            lambda: get_dataset_dicts(img_dir, annotation_file, is_train=False))
    MetadataCatalog.get(dataset_test_name).set(
        thing_classes=class_names,
        evaluator_type='coco'
    )
    
    return dataset_train_name, dataset_test_name, num_classes

def build_config(model_zoo_config_name, dataset_train_name, dataset_test_name, 
                 output_dir, num_classes, prediction_score_threshold=0.7, 
                 base_lr=0.0005, max_iter=5000, batch_size=128):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_config_name)
    
    cfg.DATASETS.TRAIN = (dataset_train_name,)
    cfg.DATASETS.TEST = (dataset_test_name,)
    
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
    
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = 'cpu'  
    os.makedirs(output_dir, exist_ok=True)
    
    return cfg

def main():
    img_dir = './images'  
    annotation_file = './samples.json' 
    
    model_zoo_config_name = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    trained_model_output_dir = './output'
    prediction_score_threshold = 0.7
    base_lr = 0.0005
    max_iter = 5000
    batch_size = 128
    
    dataset_train_name, dataset_test_name, num_classes = register_dataset(
        img_dir, annotation_file
    )
    
    cfg = build_config(
        model_zoo_config_name, 
        dataset_train_name, 
        dataset_test_name, 
        trained_model_output_dir, 
        num_classes,
        prediction_score_threshold, 
        base_lr, 
        max_iter, 
        batch_size
    )
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
