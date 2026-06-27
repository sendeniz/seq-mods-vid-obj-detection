import cv2 as cv
import numpy as np
import torch
import torch.optim as optim
from utils.utils import non_max_suppression, cells_to_bboxes
from models.holov4_enas import HoloV4_Enas_EfficentNet
from utils.utils import draw_bounding_box_vid
from utils.augmentations import AugmentImage
import time
import os 

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cv.setNumThreads(2)
torch.set_num_threads(2)
# OMP_NUM_THREADS=2 used in terminal

def main():
    S = [19, 38, 76]
    anchors = [
        [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
        [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
        [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
    ]

    scaled_anchors = torch.tensor(anchors) / (1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
    
    gate = "none"
    conf_thresh = 0.57
    nms_iou_thresh = 0.5
    nclasses = 30
    lr = 0.00001
    weight_decay = 0.0
    path_cpt_file = 'cpts/yolov4_608_run_0.cpt'
    
    loaded_checkpoint = torch.load(path_cpt_file, map_location=device)
    model = HoloV4_Enas_EfficentNet(hidden_size=1024, maxlength=200, nclasses=nclasses, gate=gate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.load_state_dict(loaded_checkpoint["model_state_dict"])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    
    print("Pretrained YoloV Enas 608 EfficientNet S Net initialized.")
    
    model.eval()
    
    # Path to the video file
    video_path = 'application/test_00076005.mp4'
    output_path = 'application/yolo_conf57.mp4'
    
    cap = cv.VideoCapture(video_path)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_rec = cv.VideoWriter(output_path, fourcc, 30, (608, 608))
    
    fps_start = time.time()
    prev = fps_start
    
    t = 0 
    carry = ((None, None), (None, None), (None, None), (None, None))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.resize(frame, (608, 608))  # Resize frame to input size
        input_frame = frame.astype(np.float32)  # Convert frame to float32
        input_frame = AugmentImage.normalize(input_frame)  # Normalize frame if necessary
        input_frame = torch.tensor(input_frame).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # Convert to tensor
        
        # Calculate FPS
        fps_end = time.time()
        time_diff = fps_end - prev
        fps = int(1 / time_diff)
        prev = fps_end
        
        # Draw FPS on the frame
        fps_txt = "Yolo FPS: {}".format(fps)
        height, width = frame.shape[:2]
        frame = cv.putText(frame, fps_txt, (width - 140, 20), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
        
        # Perform inference
        with torch.no_grad():
            preds, carry = model(input_frame, t, carry)
        
        t += 1
        anchor = torch.tensor([*anchors[0]]).to(device) * preds.shape[2]
        boxes = cells_to_bboxes(preds.to(device), is_preds=True, S=preds.shape[2], anchors=anchor.to(device))[0]
        boxes = non_max_suppression(boxes, iou_threshold=nms_iou_thresh, confidence_threshold=conf_thresh)
        
        # Draw bounding boxes on the frame
        frame = draw_bounding_box_vid(frame, boxes)
        frame = frame.astype(np.uint8)
        
        # Write frame to output video
        video_rec.write(frame)
        
    # Release resources
    video_rec.release()
    cap.release()
    print("Video processed in real-time and saved.")

if __name__ == "__main__":
    main()