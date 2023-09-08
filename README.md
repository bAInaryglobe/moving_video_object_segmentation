# Moving Object Detection and Segmentation
- We have used YOLOv8 for object detection, currently, we have limited the object detection to persons only
- Using the Bounding boxes as a prompt to Segment anything model, we have generated the segmentation masks
- We have created a pipeline to combine both the models in one flow


## Usage
- We have given a Jupyter Notebook file for the pipeline, ```moving_object_detection.ipynb```
- We have also given a Python script file for easy execution:
  ```bash
  python yolo_sam.py
  ```




## Acknowledgement
Thanks [SAM](https://github.com/facebookresearch/segment-anything) and YOLOv8 for their public code and released models.
