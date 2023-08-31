import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

from ultralytics import YOLO
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

## checkpoints for sam
sam_checkpoints = "checkpoints"
vit_h = "sam_vit_h_4b8939.pth"
vit_b = "sam_vit_b_01ec64.pth"
vit_l = "sam_vit_l_0b3195.pth"

## check for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_frame(frame, yolo_model, predictor):
    
    results = yolo_model(frame, conf=0.25, classes=[0])
    
    ## Process results
    for result in results:
        boxes = result.boxes
        
    bbox = boxes.xyxy
    #confidences = boxes.conf
    #classes = boxes.cls 
    #predictor = SamPredictor(sam)
    predictor.set_image(frame)
    
    input_boxes = bbox.to(predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
    
    masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
    )
    
    return masks

def mask2img(mask):
    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (0, 255, 255),
    }
    
    palette_tensor = torch.tensor([palette[x] for x in mask.flatten()], dtype=torch.uint8)
    image = palette_tensor.reshape(mask.shape[0], mask.shape[1], 3)
    return image

def show_mask(masks):
    mask_images = [mask2img(torch.squeeze(mask).cpu().numpy()) for mask in masks]
    combined_mask = torch.sum(torch.stack(mask_images, dim=0), dim=0)
    return combined_mask.detach().cpu().numpy()


if __name__ == '__main__':
    ## Load YOLO
    ## yolo model
    yolo_model = YOLO('yolov8n.pt').to(device)

    ## sam model
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=os.path.join(sam_checkpoints, vit_l))
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    
    
    
    ## Load video
    video_path  = os.path.join(os.getcwd(), 'src','classroom.mp4')
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture(0)
    
    if cap.isOpened() == False:
        print("Error in loading the video")
    
    i = 0
    
    # # Get the video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # # Define the output video path
    output_path = os.path.join(os.getcwd(), 'output', 'processed_video.mp4')

    # # Create a VideoWriter object to save the processed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while(cap.isOpened()):
        ret, frame = cap.read()
    
        masks = process_frame(frame, yolo_model, predictor)
        colour_mask = show_mask(masks)
        
        #dispaly frame and colour mask in same window
        frame = ((frame/np.max(frame))*255).astype(np.uint8)
        colour_mask = cv2.addWeighted(colour_mask.astype(np.uint8), 0.3, frame, 0.7, 0, dtype=cv2.CV_8U)#colour_mask.astype(np.uint8))
        #cv2.imshow('frame', frame)
        cv2.imshow('frame', colour_mask)
        
        # Write the combined frame to the output video
        out.write(colour_mask)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
        i = i + 1
        ## save frame and make video
       

    cap.release()
    cv2.destroyAllWindows()