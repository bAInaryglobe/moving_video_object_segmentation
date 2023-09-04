# import os

# import torch
# from ultralytics import YOLO



# if __name__ == '__main__':
#     video_path = os.path.join('src', 'classroom.mp4')
    
#     ## check for device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     yolo_model = YOLO('yolov8x-seg.pt', task='segment',).to(device)
    
#     predictions = yolo_model(video_path, conf=0.25, classes=[0], save_dir='output')
    
 
    
    
    