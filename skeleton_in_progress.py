import os
import csv
import json
import sys
import cv2
import datetime
import logging
from threading import Thread
import time

import pandas as pd
import mediapipe as mp
import numpy as np

csv_config_path = sys.argv[1]
video_path = sys.argv[2]
csv_coords_dir = sys.argv[3]

congif_df = pd.read_csv(csv_config_path)    
start_time, end_time, labels = congif_df['start'], congif_df['end'], congif_df['action']

def multi():
    pass

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(video_path)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
seconds = round(frames / fps)
video_time = datetime.timedelta(seconds=seconds)
print(frames, fps, seconds)

#t = Thread(target=speak, args=(a,))
#t.start()

for start, end, label in zip(start_time, end_time, labels):
    idx = round(start,1)*round(fps,0)
    #idx = -1
    LS_x, RS_x, LE_x, RE_x, LW_x, RW_x, LP_x, RP_x, LI_x, RI_x, LT_x, RT_x, LH_x, RH_x, LK_x, RK_x, LHE_x, RHE_x, LF_x, RF_x\
    = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    LS_y, RS_y, LE_y, RE_y, LW_y, RW_y, LP_y, RP_y, LI_y, RI_y, LT_y, RT_y, LH_y, RH_y, LK_y, RK_y, LHE_y, RHE_y, LF_y, RF_y\
    = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    LS_z, RS_z, LE_z, RE_z, LW_z, RW_z, LP_z, RP_z, LI_z, RI_z, LT_z, RT_z, LH_z, RH_z, LK_z, RK_z, LHE_z, RHE_z, LF_z, RF_z\
    = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    label_list = []
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        #while cap.isOpened():
        while True:
            idx += 1
            success, image = cap.read()
            #success, image = Thread(target = cap.read, daemon=True).start()
            if not success:
                print("Recapturing the video.")
                #break
                cap = cv2.VideoCapture(video_path)
                continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # Flip the image horizontally for a selfie-view display.
            
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        #if idx >= round(start,1)*fps:
            print("index of frame:",idx)
            keypoints = pose.process(image)
            lm = keypoints.pose_landmarks
            lmPose  = mp_pose.PoseLandmark
            if type(lm) == type(None):
                print("No pose detected")
                continue
            label_list.append(label)
            LS_x.append(lm.landmark[lmPose.LEFT_SHOULDER].x), RS_x.append(lm.landmark[lmPose.RIGHT_SHOULDER].x)
            LS_y.append(lm.landmark[lmPose.LEFT_SHOULDER].y), RS_y.append(lm.landmark[lmPose.RIGHT_SHOULDER].y)
            LS_z.append(lm.landmark[lmPose.LEFT_SHOULDER].z), RS_z.append(lm.landmark[lmPose.RIGHT_SHOULDER].z)
            LE_x.append(lm.landmark[lmPose.LEFT_ELBOW].x), RE_x.append(lm.landmark[lmPose.RIGHT_ELBOW].x)
            LE_y.append(lm.landmark[lmPose.LEFT_ELBOW].y), RE_y.append(lm.landmark[lmPose.RIGHT_ELBOW].y)
            LE_z.append(lm.landmark[lmPose.LEFT_ELBOW].z), RE_z.append(lm.landmark[lmPose.RIGHT_ELBOW].z)
            LW_x.append(lm.landmark[lmPose.LEFT_WRIST].x), RW_x.append(lm.landmark[lmPose.RIGHT_WRIST].x)
            LW_y.append(lm.landmark[lmPose.LEFT_WRIST].y), RW_y.append(lm.landmark[lmPose.RIGHT_WRIST].y)
            LW_z.append(lm.landmark[lmPose.LEFT_WRIST].z), RW_z.append(lm.landmark[lmPose.RIGHT_WRIST].z)
            LP_x.append(lm.landmark[lmPose.LEFT_PINKY].x), RP_x.append(lm.landmark[lmPose.RIGHT_PINKY].x)
            LP_y.append(lm.landmark[lmPose.LEFT_PINKY].y), RP_y.append(lm.landmark[lmPose.RIGHT_PINKY].y)
            LP_z.append(lm.landmark[lmPose.LEFT_PINKY].z), RP_z.append(lm.landmark[lmPose.RIGHT_PINKY].z)
            LI_x.append(lm.landmark[lmPose.LEFT_INDEX].x), RI_x.append(lm.landmark[lmPose.RIGHT_INDEX].x)
            LI_y.append(lm.landmark[lmPose.LEFT_INDEX].y), RI_y.append(lm.landmark[lmPose.RIGHT_INDEX].y)
            LI_z.append(lm.landmark[lmPose.LEFT_INDEX].z), RI_z.append(lm.landmark[lmPose.RIGHT_INDEX].z)
            LT_x.append(lm.landmark[lmPose.LEFT_THUMB].x), RT_x.append(lm.landmark[lmPose.RIGHT_THUMB].x)
            LT_y.append(lm.landmark[lmPose.LEFT_THUMB].y), RT_y.append(lm.landmark[lmPose.RIGHT_THUMB].y)
            LT_z.append(lm.landmark[lmPose.LEFT_THUMB].z), RT_z.append(lm.landmark[lmPose.RIGHT_THUMB].z)
            LH_x.append(lm.landmark[lmPose.LEFT_HIP].x), RH_x.append(lm.landmark[lmPose.RIGHT_HIP].x)
            LH_y.append(lm.landmark[lmPose.LEFT_HIP].y), RH_y.append(lm.landmark[lmPose.RIGHT_HIP].y)
            LH_z.append(lm.landmark[lmPose.LEFT_HIP].z), RH_z.append(lm.landmark[lmPose.RIGHT_HIP].z)
            LK_x.append(lm.landmark[lmPose.LEFT_KNEE].x), RK_x.append(lm.landmark[lmPose.RIGHT_KNEE].x)
            LK_y.append(lm.landmark[lmPose.LEFT_KNEE].y), RK_y.append(lm.landmark[lmPose.RIGHT_KNEE].y)
            LK_z.append(lm.landmark[lmPose.LEFT_KNEE].z), RK_z.append(lm.landmark[lmPose.RIGHT_KNEE].z)
            LHE_x.append(lm.landmark[lmPose.LEFT_HEEL].x), RHE_x.append(lm.landmark[lmPose.RIGHT_HEEL].x)
            LHE_y.append(lm.landmark[lmPose.LEFT_HEEL].y), RHE_y.append(lm.landmark[lmPose.RIGHT_HEEL].y)
            LHE_z.append(lm.landmark[lmPose.LEFT_HEEL].z), RHE_z.append(lm.landmark[lmPose.RIGHT_HEEL].z)
            LF_x.append(lm.landmark[lmPose.LEFT_FOOT_INDEX].x), RF_x.append(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x)
            LF_y.append(lm.landmark[lmPose.LEFT_FOOT_INDEX].y), RF_y.append(lm.landmark[lmPose.RIGHT_FOOT_INDEX].y)
            LF_z.append(lm.landmark[lmPose.LEFT_FOOT_INDEX].z), RF_z.append(lm.landmark[lmPose.RIGHT_FOOT_INDEX].z)
            if idx >= round(end,1)*fps:
                print("extracting one action ended")
                coords_df = pd.DataFrame(
                    {   'LS_x': LS_x,'RS_x': RS_x,'LS_y': LS_y,'RS_y': RS_y,'LS_z': LS_z,'RS_z': RS_z,              #shoulder
                        'LE_x': LE_x,'RE_x': RE_x,'LE_y': LE_y,'RE_y': RE_y,'LE_z': LE_z,'RE_z': RE_z,              #elbow
                        'LW_x': LW_x,'RW_x': RW_x,'LW_y': LW_y,'RW_y': RW_y,'LW_z': LW_z,'RW_z': RW_z,              #wrist
                        'LP_x': LP_x,'RP_x': RP_x,'LP_y': LP_y,'RP_y': RP_y,'LP_z': LP_z,'RP_z': RP_z,              #pinkie
                        'LI_x': LI_x,'RI_x': RI_x,'LI_y': LI_y,'RI_y': RI_y,'LI_z': LI_z,'RI_z': RI_z,              #index
                        'LT_x': LT_x,'RT_x': RT_x,'LT_y': LT_y,'RT_y': RT_y,'LT_z': LT_z,'RT_z': RT_z,              #thumb
                        'LH_x': LH_x,'RH_x': RH_x,'LH_y': LH_y,'RH_y': RH_y,'LH_z': LH_z,'RH_z': RH_z,              #hip
                        'LK_x': LF_x,'RK_x': RK_x,'LK_y': LF_y,'RK_y': RK_y,'LK_z': LF_z,'RK_z': RK_z,              #knee
                        'LHE_x': LHE_x,'RHE_x': RHE_x,'LHE_y': LHE_y,'RHE_y': RHE_y,'LHE_z': LHE_z,'RHE_z': RHE_z,  #heel
                        'LF_x': LF_x,'RF_x': RF_x,'LF_y': LF_y,'RF_y': RF_y,'LF_z': LF_z,'RF_z': RF_z,              #foot
                        'label': label_list
                    }
                )
                #outdir = './dir'
                #if not os.path.exists(outdir):
                #    os.mkdir(outdir)
                #fullname = os.path.join(outdir, outname)
                coords_df.to_csv(csv_coords_dir + '/'\
                                    + os.path.splitext(os.path.basename(video_path))[0]\
                                    + '_' + str(start) + '.csv',\
                                    index = False)#, columns=['Titles', 'Emails', 'Links'])
                break
 
cap.release()
