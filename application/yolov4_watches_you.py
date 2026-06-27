import cv2 as cv
import numpy as np
import time 
import torch
import torch.optim as optim
import pyrealsense2 as rs  # Import RealSense SDK
from utils.utils import non_max_suppression, cells_to_bboxes
from models.yolov4 import YoloV4_EfficentNet
from utils.utils import draw_bounding_box
from utils.augmentations import AugmentImage

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
S = [19, 38, 76]
anchors = [
       [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
       [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
       [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
       ]

scaled_anchors = torch.tensor(anchors) / ( 1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) )

conf_thresh = 0.8
nms_iou_thresh = 0.5
map_iou_thresh = 0.5
nworkers = 2
nclasses = 80
lr =  0.00001
weight_decay = 0.0005
path_cpt_file = f'cpts/yolov4_608_mscoco.cpt'
checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))
model = YoloV4_EfficentNet(nclasses = nclasses).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()
print("Petrained YoloV4 608 EfficentNet S Net initalized.")


# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # Configure color stream

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Check if color frame is valid
        if not color_frame:
            continue

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Process the frame
        frame = cv.resize(color_image, (608, 608)).astype(np.float32)
        frame = AugmentImage.normalize(frame)
        frame = torch.tensor(frame).unsqueeze(0).permute(0, 3, 1, 2).to(device)

        # Perform inference
        with torch.no_grad():
            preds = model(frame)
        
        boxes = [] 
        for i in range(preds[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes = cells_to_bboxes(preds[i], is_preds=True, S=preds[i].shape[2], anchors = anchor)[0]
        boxes = non_max_suppression(boxes, iou_threshold = nms_iou_thresh, confidence_threshold = conf_thresh)

        frame = draw_bounding_box(frame[0].permute(1, 2, 0).to("cpu") * 255, boxes)
        frame = cv.cvtColor(frame.astype(np.uint8), cv.COLOR_BGR2RGB)
        # Display the frame
        cv.imshow('RealSense Video', frame)

        # Wait for a key press; exit loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv.destroyAllWindows()
