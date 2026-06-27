import cv2 as cv
import numpy as np
import torch
import torch.optim as optim
from utils.utils import non_max_suppression, cells_to_bboxes
from models.yolov4 import YoloV4_EfficentNet
from utils.utils import draw_bounding_box
from utils.augmentations import AugmentImage
import time
cv.setNumThreads(1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
S = [19, 38, 76]
anchors = [
       [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
       [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
       [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
       ]

scaled_anchors = torch.tensor(anchors) / (1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

conf_thresh = 0.7 # for yolo use .70
nms_iou_thresh = 0.5
map_iou_thresh = 0.5
nworkers = 2
nclasses = 80
lr = 0.00001
weight_decay = 0.0005
path_cpt_file = 'cpts/yolov4_608_mscoco.cpt'
checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))
model = YoloV4_EfficentNet(nclasses=nclasses).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()
print("Pretrained YoloV4 608 EfficientNet S Net initialized.")

# Path to the video file
video_path = 'application/demo.mp4'
output_path = 'application/demo_processed.mp4'

cap = cv.VideoCapture(video_path)

fps = 0
fps_start = 0
prev = 0 

fourcc = cv.VideoWriter_fourcc(*'mp4v')
video_rec = cv.VideoWriter(output_path, 
                         fourcc,
                         30, (608, 608))
# Initialize variables for FPS calculation
fps_start = time.time()
prev = fps_start

c = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.array(frame)
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
    fps_txt = "FPS: {}".format(fps)
    height, width = frame.shape[:2]
    frame = cv.putText(frame, fps_txt, (width - 80, 20), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    
    # Perform inference
    with torch.no_grad():
        preds = model(input_frame)
        
    boxes = []
    for i in range(preds[0].shape[1]):
        anchor = scaled_anchors[i]
        boxes = cells_to_bboxes(preds[i].to(device), is_preds=True, S=preds[i].shape[2], anchors=anchor.to(device))[0]
    
    boxes = non_max_suppression(boxes, iou_threshold=nms_iou_thresh, confidence_threshold=conf_thresh)
    
    # Draw bounding boxes on the frame
    frame = draw_bounding_box(frame, boxes)
    frame = frame.astype(np.uint8)
    
    # Write frame to output video
    #cv.imwrite(f'application/img_{c}.png', frame)
    video_rec.write(frame)
    #c = c + 1
    # Uncomment if you want to exit by pressing 'q'
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
video_rec.release()
cap.release()
# cv.destroyAllWindows()  # Uncomment if you need to close all OpenCV windows

print("Video processed in real time and saved.")