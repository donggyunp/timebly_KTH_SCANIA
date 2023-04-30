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

from skeleton_to_csv import extract_skeleton

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
    idx = round(start,1)*fps
    #idx = -1
    LS, RS, LE, RE, LW, RW, LP, RP, LI, RI, LT, RT, LH, RH, LK, RK, LHE, RHE, LF, RF\
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
            label_list.append(label)
            LS.append(lm.landmark[lmPose.LEFT_SHOULDER].x), RS.append(lm.landmark[lmPose.RIGHT_SHOULDER].x)
            LE.append(lm.landmark[lmPose.LEFT_ELBOW].x), RE.append(lm.landmark[lmPose.RIGHT_ELBOW].x)
            LW.append(lm.landmark[lmPose.LEFT_WRIST].x), RW.append(lm.landmark[lmPose.RIGHT_WRIST].x)
            LP.append(lm.landmark[lmPose.LEFT_PINKY].x), RP.append(lm.landmark[lmPose.RIGHT_PINKY].x)
            LI.append(lm.landmark[lmPose.LEFT_INDEX].x), RI.append(lm.landmark[lmPose.RIGHT_INDEX].x)
            LT.append(lm.landmark[lmPose.LEFT_THUMB].x), RT.append(lm.landmark[lmPose.RIGHT_THUMB].x)
            LH.append(lm.landmark[lmPose.LEFT_HIP].x), RH.append(lm.landmark[lmPose.RIGHT_HIP].x)
            LK.append(lm.landmark[lmPose.LEFT_KNEE].x), RK.append(lm.landmark[lmPose.RIGHT_KNEE].x)
            LHE.append(lm.landmark[lmPose.LEFT_HEEL].x), RHE.append(lm.landmark[lmPose.RIGHT_HEEL].x)
            LF.append(lm.landmark[lmPose.LEFT_FOOT_INDEX].x), RF.append(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x)
            if idx >= round(end,1)*fps:
                print("extracting one action ended")
                coords_df = pd.DataFrame(
                    {'LS': LS,'RS': RS,#shoulder
                        'LE': LE,'RE': RE,#elbow
                        'LW': LW,'RW': RW,#wrist
                        'LP': LP,'RP': RP,#pinkie
                        'LI': LI,'RI': RI,#index
                        'LT': LT,'RT': RT,#thumb
                        'LH': LH,'RH': RH,#hip
                        'LK': LF,'RK': RK,#knee
                        'LHE': LHE,'RHE': RHE,#heel
                        'LF': LF,'RF': RF,#foot
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
