import cv2
import argparse
import os
import random
import torch
from torchvision import datasets, transforms
import numpy as np
from model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color
import csv

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='Input video path. Live cam is used when not specified')
parser.add_argument('--model_weight', type=str, help='Path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--jitter', type=int, help='Jitter bbox n times, and average results', default=0)
parser.add_argument('--save_vis', help='Saves output as video', action='store_true')
parser.add_argument('--save_csv', help='Saves output as CSV', action='store_true')
parser.add_argument('--display_off', help='Do not display frames', action='store_true')

args = parser.parse_args()

# BBox jitter function
def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right + bbox_left) / 2.0
    cy = (bbox_bottom + bbox_top) / 2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right - cx) * scale + cx
    bbox_left = (bbox_left - cx) * scale + cx
    bbox_top = (bbox_top - cy) * scale + cy
    bbox_bottom = (bbox_bottom - cy) * scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom

# Function to draw rectangle
def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

# Load OpenCV face detection model
def load_opencv_face_detector():
    net = cv2.dnn.readNetFromTensorflow('models/opencv_face_detector_uint8.pb', 'models/opencv_face_detector.pbtxt')
    return net

# Face detection with OpenCV
def detect_faces_opencv(frame, net):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    bbox = []
    height, width = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (left, top, right, bottom) = box.astype("int")
            bbox.append([left, top, right, bottom])

    return bbox

# Main run function
def run(video_path, model_weight, jitter, save_vis, display_off, save_csv):
    red = Color("red")
    colors = list(red.range_to(Color("green"), 10))
    font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 40)

    # Video capture
    if video_path is None:
        cap = cv2.VideoCapture(0)
        video_path = 'live.avi'
    else:
        cap = cv2.VideoCapture(video_path)

    csv_file = None
    csv_writer = None
    outvid = None

    # Prepare CSV file for output
    if save_csv:
        csv_file_name = os.path.splitext(os.path.basename(video_path))[0] + '_output.csv'
        csv_file = open(csv_file_name, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'Left', 'Top', 'Right', 'Bottom', 'Probability'])
        print(f"Saving CSV output to {csv_file_name}")

    if save_vis:
        outvis_name = os.path.splitext(os.path.basename(video_path))[0] + '_output.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        outvid = cv2.VideoWriter(outvis_name, fourcc, fps, (width, height))
        print(f"Saving video output to {outvis_name}")

    # Load face detector model
    face_net = load_opencv_face_detector()

    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    frame_cnt = 0

    # Define transformations
    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the model
    model = model_static(model_weight)
    model.eval()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width, channels = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_cnt += 1

            # Detect faces using OpenCV
            bboxes = detect_faces_opencv(frame, face_net)

            # Write empty row to CSV if no faces detected
            if save_csv and csv_writer is not None and not bboxes:
                csv_writer.writerow([frame_cnt, "", "", "", "", ""])

            # Convert frame to PIL image for drawing
            frame_pil = Image.fromarray(rgb_frame)

            for b in bboxes:
                face = frame_pil.crop((b[0], b[1], b[2], b[3]))
                img = test_transforms(face)
                img.unsqueeze_(0)

                # Apply jitter if needed
                if jitter > 0:
                    for i in range(jitter):
                        bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                        bj = [bj_left, bj_top, bj_right, bj_bottom]
                        facej = frame_pil.crop((bj))
                        img_jittered = test_transforms(facej)
                        img_jittered.unsqueeze_(0)
                        img = torch.cat([img, img_jittered])

                # Model inference
                with torch.no_grad():
                    output = model(img)

                if jitter > 0:
                    output = torch.mean(output, 0)
                score = torch.sigmoid(output).item()

                # Draw bounding box and score
                coloridx = min(int(round(score * 10)), 9)
                draw = ImageDraw.Draw(frame_pil)
                drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
                draw.text((b[0], b[3]), str(round(score, 2)), fill=(255, 255, 255, 128), font=font)

                # Write to CSV file
                if save_csv and csv_writer is not None:
                    csv_writer.writerow([frame_cnt, f"{b[0]:.4f}", f"{b[1]:.4f}", f"{b[2]:.4f}", f"{b[3]:.4f}",
                                         f"{round(score, 4):.4f}"])

            # Convert PIL image back to OpenCV format
            frame_np = np.asarray(frame_pil)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # Save the frame if video saving is enabled
            if save_vis and outvid is not None:
                outvid.write(frame_bgr)

            # Display the frame if not disabled
            if not display_off:
                cv2.imshow('', frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    # Release resources
    if save_vis and outvid is not None:
        outvid.release()
    if save_csv and csv_file is not None:
        csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print('DONE!')

if __name__ == "__main__":
    run(args.video, args.model_weight, args.jitter, args.save_vis, args.display_off, args.save_csv)