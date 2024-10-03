import csv
import cv2
import numpy as np
from collections import defaultdict
import os
import argparse

def read_gaze_data(gaze_csv_path):
    gaze_data = {}
    with open(gaze_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            timestamp = float(row['timestamp'])

            try:
                # Parse the gaze2d string safely
                gaze_str = row['gaze2d'].strip('[]" ')
                gaze_x, gaze_y = map(float, gaze_str.split(','))

                gaze_data[frame] = {
                    'timestamp': timestamp,
                    'gaze_x': gaze_x,
                    'gaze_y': gaze_y
                }
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse gaze2d data in frame {frame}: {row['gaze2d']}")
                gaze_data[frame] = {
                    'timestamp': timestamp,
                    'gaze_x': np.nan,
                    'gaze_y': np.nan
                }
    return gaze_data


def read_face_detection_data(face_csv_path):
    face_data = {}
    with open(face_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row['Frame'])
                # Convert bounding box coordinates to normalized values (0-1 range)
                left = float(row['Left'])
                top = float(row['Top'])
                right = float(row['Right'])
                bottom = float(row['Bottom'])
                probability = float(row['Probability'])

                face_data[frame] = {
                    'bbox': (left, top, right, bottom),
                    'probability': probability
                }
            except ValueError as e:
                print(f"Warning: Could not parse face detection data in row: {row}. Error: {e}")
                continue
    return face_data


def is_gaze_in_face(gaze_x, gaze_y, face_bbox, video_width, video_height):
    left, top, right, bottom = face_bbox
    # Convert gaze coordinates to pixel values
    gaze_x_pixel = gaze_x * video_width
    gaze_y_pixel = gaze_y * video_height
    return left <= gaze_x_pixel <= right and top <= gaze_y_pixel <= bottom


def process_data(gaze_data, face_data, output_csv_path, video_width, video_height):
    combined_data = defaultdict(lambda: {
        'timestamp': np.nan,
        'gaze_x': np.nan,
        'gaze_y': np.nan,
        'eye_contact': np.nan,
        'mutual_eye_contact': np.nan,
        'face_bbox': (np.nan, np.nan, np.nan, np.nan),
        'probability': np.nan
    })

    all_frames = set(list(gaze_data.keys()) + list(face_data.keys()))

    for frame in all_frames:
        gaze_info = gaze_data.get(frame, {})
        face_info = face_data.get(frame, {})

        data = combined_data[frame]

        # Update with gaze data if available
        if gaze_info:
            data.update({
                'timestamp': gaze_info['timestamp'],
                'gaze_x': gaze_info['gaze_x'],
                'gaze_y': gaze_info['gaze_y']
            })

        # Update with face data if available
        if face_info:
            data.update({
                'face_bbox': face_info['bbox'],
                'probability': face_info['probability'],
                'eye_contact': face_info['probability']
            })

        # Calculate mutual eye contact
        if not np.isnan(data['gaze_x']) and not np.isnan(data['face_bbox'][0]):
            data['mutual_eye_contact'] = (data['eye_contact'] > 0.5 and
                                          is_gaze_in_face(data['gaze_x'], data['gaze_y'],
                                                          data['face_bbox'], video_width, video_height))
        else:
            data['mutual_eye_contact'] = np.nan

    # Write to CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'timestamp', 'gaze_x', 'gaze_y', 'eye_contact', 'mutual_eye_contact'])

        for frame, data in sorted(combined_data.items()):
            writer.writerow([
                frame,
                data['timestamp'],
                data['gaze_x'],
                data['gaze_y'],
                data['eye_contact'],
                data['mutual_eye_contact']
            ])

    return combined_data


def process_video(input_video_path, output_video_path, combined_data):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in combined_data:
            data = combined_data[frame_number]

            # Draw face bounding box
            if not np.isnan(data['face_bbox'][0]):
                bbox = data['face_bbox']
                # Change bounding box color based on mutual eye contact
                if isinstance(data['mutual_eye_contact'], bool) and data['mutual_eye_contact']:
                    box_color = (0, 255, 0)  # Green for mutual eye contact
                else:
                    box_color = (255, 0, 0)  # Red for no mutual eye contact

                cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              box_color, 2)

            # Draw gaze point
            if not np.isnan(data['gaze_x']) and not np.isnan(data['gaze_y']):
                gaze_x_pixel = int(data['gaze_x'] * width)
                gaze_y_pixel = int(data['gaze_y'] * height)
                # Change gaze point color based on mutual eye contact
                if isinstance(data['mutual_eye_contact'], bool) and data['mutual_eye_contact']:
                    gaze_color = (0, 255, 0)  # Green for mutual eye contact
                else:
                    gaze_color = (0, 0, 255)  # Red for no mutual eye contact
                cv2.circle(frame, (gaze_x_pixel, gaze_y_pixel), 10, gaze_color, -1)

            # Add text for eye contact probability
            if not np.isnan(data['eye_contact']):
                eye_contact_text = f"Eye Contact Prob: {data['eye_contact']:.4f}"
                cv2.putText(frame, eye_contact_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Add text for mutual eye contact
            mutual_text = "Mutual: "
            if isinstance(data['mutual_eye_contact'], bool):
                mutual_text += str(data['mutual_eye_contact'])
                text_color = (0, 255, 0) if data['mutual_eye_contact'] else (0, 0, 255)
            else:
                mutual_text += "NaN"
                text_color = (255, 255, 255)

            cv2.putText(frame, mutual_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Process eye contact data")
    parser.add_argument("--gaze_csv", required=True, help="Path to the gaze CSV file")
    parser.add_argument("--face_csv", required=True, help="Path to the face detection CSV file")
    parser.add_argument("--input_video", required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", required=True, help="Directory for output files")
    parser.add_argument("--generate_csv", type=str, choices=['true', 'false'], default='true',
                        help="Whether to generate CSV output")
    parser.add_argument("--generate_video", type=str, choices=['true', 'false'], default='true',
                        help="Whether to generate video output")

    args = parser.parse_args()

    try:
        # Get video dimensions
        cap = cv2.VideoCapture(args.input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {args.input_video}")
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Read data
        print("Reading gaze data...")
        gaze_data = read_gaze_data(args.gaze_csv)
        print(f"Read {len(gaze_data)} gaze data points")

        print("Reading face detection data...")
        face_data = read_face_detection_data(args.face_csv)
        print(f"Read {len(face_data)} face detection data points")

        # Process data
        output_csv_path = os.path.join(args.output_dir, 'final_output.csv')
        combined_data = process_data(gaze_data, face_data, output_csv_path, video_width, video_height)

        # Generate outputs based on flags
        if args.generate_video.lower() == 'true':
            print("Processing video...")
            output_video_path = os.path.join(args.output_dir, 'final_output_video.mp4')
            process_video(args.input_video, output_video_path, combined_data)

        print("Processing complete!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()