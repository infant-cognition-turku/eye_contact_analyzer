import cv2
import json
import pandas as pd
import os
import gzip
import shutil
import ast
import numpy as np
import argparse


def gunzip_file(file_path, output_path):
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def process_gaze_data(gz_file_path, decompressed_dir):
    os.makedirs(decompressed_dir, exist_ok=True)

    # Define the output file name for the decompressed JSON
    output_file_name = 'gazedata.json'
    output_path = os.path.join(decompressed_dir, output_file_name)

    # Decompress the .gz file
    gunzip_file(gz_file_path, output_path)
    print(f"Decompressed {gz_file_path} to {output_path}")

    # Load and process the JSON data
    data = []
    with open(output_path, 'r') as file:
        for line in file:
            record = json.loads(line.strip())
            if record.get('type') == 'gaze':
                data.append({
                    'timestamp': record.get('timestamp'),
                    'gaze2d': record['data'].get('gaze2d'),
                    'gaze3d': record['data'].get('gaze3d'),
                    'eyeleft_gazeorigin': record['data'].get('eyeleft', {}).get('gazeorigin'),
                    'eyeleft_gazedirection': record['data'].get('eyeleft', {}).get('gazedirection'),
                    'eyeleft_pupildiameter': record['data'].get('eyeleft', {}).get('pupildiameter'),
                    'eyeright_gazeorigin': record['data'].get('eyeright', {}).get('gazeorigin'),
                    'eyeright_gazedirection': record['data'].get('eyeright', {}).get('gazedirection'),
                    'eyeright_pupildiameter': record['data'].get('eyeright', {}).get('pupildiameter')
                })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Define how to combine rows (same as before)
    def combine_rows(group):
        combined = group.iloc[0].copy()
        combined['frame'] = group.index[0] // 2 + 1
        for col in group.columns:
            if col != 'timestamp':
                values = group[col].dropna().values
                if len(values) > 0:
                    if isinstance(values[0], (list, tuple)):
                        # Handle list-type data
                        combined[col] = np.mean([v for v in values if v], axis=0).tolist()
                    elif pd.api.types.is_numeric_dtype(group[col]):
                        # Handle numeric data
                        combined[col] = np.mean(values)
                    else:
                        # For non-numeric, non-list data, keep the first non-null value
                        combined[col] = values[0]
                else:
                    combined[col] = np.nan
        return combined

    # Apply the combination of rows
    processed_df = df.groupby(df.index // 2).apply(combine_rows).reset_index(drop=True)

    # Reorder columns
    column_order = ['frame', 'timestamp', 'gaze2d', 'gaze3d', 'eyeleft_gazeorigin', 'eyeleft_gazedirection',
                    'eyeleft_pupildiameter', 'eyeright_gazeorigin', 'eyeright_gazedirection', 'eyeright_pupildiameter']
    processed_df = processed_df[column_order]

    return processed_df


def add_gaze_to_video(frame_data, video_file_path, output_video_path):
    cap = cv2.VideoCapture(video_file_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for _, row in frame_data.iterrows():
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame'] - 1)  # Subtract 1 because frame numbers start at 1
        ret, frame = cap.read()
        if not ret:
            break

        gaze2d = row['gaze2d']
        if isinstance(gaze2d, str):
            gaze2d = ast.literal_eval(gaze2d)

        if gaze2d and len(gaze2d) == 2:
            gaze_x, gaze_y = gaze2d
            gaze_x = int(gaze_x * frame_width)
            gaze_y = int(gaze_y * frame_height)
            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved to {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Process gaze data and optionally create a video with gaze overlay.")
    parser.add_argument("--input", help="Input .gz file path", required=True)
    parser.add_argument("--create_video", action="store_true", help="Create video with gaze overlay")

    args = parser.parse_args()

    gz_file_path = args.input
    decompressed_dir = os.path.dirname(gz_file_path)  # Output decompressed JSON in the same directory
    video_file_path = os.path.join(decompressed_dir, 'scenevideo.mp4')
    output_video_path = os.path.join(decompressed_dir, 'scenevideo_with_gaze.mp4')
    csv_output_path = os.path.join(decompressed_dir, 'gazedata_frames.csv')

    # Process the gaze data from the .gz file
    processed_df = process_gaze_data(gz_file_path, decompressed_dir)

    # Save the processed data to CSV
    processed_df.to_csv(csv_output_path, index=False)
    print(f"Frame-based gaze data saved to {csv_output_path}")

    # If the user requested video creation, add gaze overlay to the video
    if args.create_video:
        add_gaze_to_video(processed_df, video_file_path, output_video_path)
    else:
        print("Video creation skipped. Use --create_video flag to create the video.")


if __name__ == "__main__":
    main()