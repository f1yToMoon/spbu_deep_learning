from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2

def convert_to_yolo_format(bbox, img_width, img_height):
    x1, y1, x2, y2 = map(float, bbox)
    
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return f"0 {x_center} {y_center} {width} {height}"

def prepare_dataset():
    images_dir = "train/images"
    annotations_dir = "train/annotations"
    yolo_labels_dir = "train/labels"  
    
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    for ann_file in os.listdir(annotations_dir):
        if not ann_file.endswith('.txt'):
            continue
            
        image_file = os.path.join(images_dir, ann_file.replace('.txt', '.jpg'))
        if not os.path.exists(image_file):
            image_file = os.path.join(images_dir, ann_file.replace('.txt', '.png'))
            
        img = cv2.imread(image_file)
        img_height, img_width = img.shape[:2]
        
        with open(os.path.join(annotations_dir, ann_file), 'r') as f:
            boxes = f.read().strip().split('\n')
        
        yolo_boxes = []
        for box in boxes:
            if box: 
                coords = box.strip().split(',')
                yolo_box = convert_to_yolo_format(coords, img_width, img_height)
                yolo_boxes.append(yolo_box)
        
        with open(os.path.join(yolo_labels_dir, ann_file), 'w') as f:
            f.write('\n'.join(yolo_boxes))
    return "train/images" 

def train_model():
    model = YOLO('yolov8nn.pt')
    
    model.train(
        data='coco.yaml',
        epochs=40,
        imgsz=640,
        batch=16,
        name='people_detection',
        cache=True,
        device='0',  
        project='runs',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        val=False,  
    )
    
    return model

def process_images(model, image_folder):
    results_dict = {}
    
    image_folder = os.path.abspath(os.path.join(image_folder, "images"))
    
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    for idx, img_file in tqdm(enumerate(image_files), total=len(image_files), desc="Predicting"):
        img_path = os.path.join(image_folder, img_file)
        
        results = model.predict(img_path, conf=0.25)  
        
        boxes = []
        for r in results:
            for box in r.boxes:
                if box.cls == 0:  
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        img_id = int(os.path.splitext(img_file)[0])
        results_dict[img_id] = boxes
        
    
    return results_dict   

import csv

def save_results(results_dict, output_file='1.csv'):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'bboxes'])
        
        for img_id in sorted(results_dict.keys()):
            boxes = results_dict[img_id]
            if boxes:
                boxes_str = f"[{', '.join([str(tuple(box)) for box in boxes])}]"
            else:
                boxes_str = "[]"
            writer.writerow([img_id, boxes_str])

def main():
    prepare_dataset()
    
    model = train_model()
    
    test_folder = "test"  
    results = process_images(model, test_folder)
    
    save_results(results)

if __name__ == "__main__":
    main()
