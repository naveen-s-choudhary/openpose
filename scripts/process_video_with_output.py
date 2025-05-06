#!/usr/bin/env python3

import sys
import os
from sys import platform
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Add OpenPose Python module path
dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, '..'))
print(root_path)
# BODY_25 keypoint names
BODY_25_KEYPOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel"
]

# Foot keypoint indices and names
FOOT_KEYPOINTS = {
    'left': {
        'indices': [19, 20, 21],  # LBigToe, LSmallToe, LHeel
        'names': ['BigToe', 'SmallToe', 'Heel']
    },
    'right': {
        'indices': [22, 23, 24],  # RBigToe, RSmallToe, RHeel
        'names': ['BigToe', 'SmallToe', 'Heel']
    }
}

try:
    # Windows Import
    if platform == "win32":
        sys.path.append(os.path.join(root_path, 'build/python/openpose/Release'))
        os.environ['PATH'] = os.path.join(root_path, 'build/bin') + ';' + os.environ['PATH']
    # Linux/Mac Import
    else:
        sys.path.append(os.path.join(root_path, 'build/python'))
        sys.path.append(os.path.join(root_path, 'build/python/openpose'))
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    print('Error: ' + str(e))
    sys.exit(-1)

import cv2

def get_main_person(keypoints):
    """Find the main person (usually the one with most visible keypoints)"""
    if keypoints is None or len(keypoints) == 0:
        return None
    
    # Calculate average confidence for each person
    avg_confidences = []
    for person in keypoints:
        # Filter out zero confidence points
        valid_points = person[person[:, 2] > 0]
        if len(valid_points) > 0:
            avg_conf = np.mean(valid_points[:, 2])
            avg_confidences.append(avg_conf)
        else:
            avg_confidences.append(0)
    
    # Return the person with highest average confidence
    return np.argmax(avg_confidences) if avg_confidences else None

def extract_foot_keypoints(frame_number, timestamp, person_keypoints):
    """Extract foot keypoints for both feet"""
    foot_data = {'frame': int(frame_number), 'timestamp': float(timestamp)}
    
    # Initialize with zeros
    for foot in ['left', 'right']:
        for name in FOOT_KEYPOINTS[foot]['names']:
            foot_data[f'{foot}_{name}_x'] = 0.0
            foot_data[f'{foot}_{name}_y'] = 0.0
            foot_data[f'{foot}_{name}_conf'] = 0.0
    
    # Extract foot keypoints if person is detected
    if person_keypoints is not None:
        for foot in ['left', 'right']:
            for idx, name in zip(FOOT_KEYPOINTS[foot]['indices'], FOOT_KEYPOINTS[foot]['names']):
                keypoint = person_keypoints[idx]
                foot_data[f'{foot}_{name}_x'] = float(keypoint[0])
                foot_data[f'{foot}_{name}_y'] = float(keypoint[1])
                foot_data[f'{foot}_{name}_conf'] = float(keypoint[2])
    
    return foot_data

def process_video(video_path, output_dir, output_video_path=None):
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = os.path.join(root_path, "models/")
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = "-1x368"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer if output path is provided
    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize DataFrame columns
    columns = ['frame', 'timestamp']
    # Add columns for each keypoint (x, y, confidence) with body part names
    for i, keypoint_name in enumerate(BODY_25_KEYPOINTS):
        columns.extend([f'{keypoint_name}_x', f'{keypoint_name}_y', f'{keypoint_name}_conf'])
    
    # Initialize lists to store data
    frame_data = []
    foot_data = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Initialize frame row with zeros
        frame_row = {'frame': int(frame_number), 'timestamp': float(frame_number / fps)}
        for keypoint_name in BODY_25_KEYPOINTS:
            frame_row[f'{keypoint_name}_x'] = 0.0
            frame_row[f'{keypoint_name}_y'] = 0.0
            frame_row[f'{keypoint_name}_conf'] = 0.0

        # Get keypoints
        if datum.poseKeypoints is not None:
            # Find main person
            main_person_idx = get_main_person(datum.poseKeypoints)
            
            if main_person_idx is not None:
                main_person = datum.poseKeypoints[main_person_idx]
                for keypoint_idx, keypoint in enumerate(main_person):
                    keypoint_name = BODY_25_KEYPOINTS[keypoint_idx]
                    frame_row[f'{keypoint_name}_x'] = float(keypoint[0])
                    frame_row[f'{keypoint_name}_y'] = float(keypoint[1])
                    frame_row[f'{keypoint_name}_conf'] = float(keypoint[2])
                
                # Extract foot keypoints
                foot_row = extract_foot_keypoints(frame_number, frame_number / fps, main_person)
                foot_data.append(foot_row)

        frame_data.append(frame_row)
        frame_number += 1
        print(f"\rProcessing frame {frame_number}/{frame_count}", end='')

        # Write frame to output video if enabled
        if video_writer is not None:
            video_writer.write(datum.cvOutputData)

    print("\nProcessing complete!")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'keypoints.csv')
    df = pd.DataFrame(frame_data, columns=columns)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV output to {csv_path}")

    # Save foot keypoints to separate CSV
    foot_csv_path = os.path.join(output_dir, 'foot_keypoints.csv')
    foot_df = pd.DataFrame(foot_data)
    foot_df.to_csv(foot_csv_path, index=False)
    print(f"Saved foot keypoints to {foot_csv_path}")

    # Save to JSON
    json_path = os.path.join(output_dir, 'keypoints.json')
    json_data = {
        'video_info': {
            'frame_count': int(frame_count),
            'fps': float(fps),
            'duration': float(duration)
        },
        'keypoints': frame_data,
        'foot_keypoints': foot_data
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON output to {json_path}")

    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output_dir", default="output", help="Directory to save the output files")
    parser.add_argument("--output_video", help="Path to save the output video with pose visualization")
    args = parser.parse_args()

    process_video(args.video_path, args.output_dir, args.output_video)

if __name__ == "__main__":
    main() 