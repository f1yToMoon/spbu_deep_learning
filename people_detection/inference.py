from ultralytics import YOLO
import os
from tqdm import tqdm
import csv

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
    model = YOLO('./runs/people_detection/weights/best.pt') 
    test_folder = "test"  
    results = process_images(model, test_folder)
    
    save_results(results)

if __name__ == "__main__":
    main()
