import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import cv2
from src.untitled_aruco_package.untitled_core import Marker, Camera, ArucoDetector, CameraCalibration, Pose, Segment, Landmark

blenderCamera = Camera(id=0, reference='Blender Camera', calibration=CameraCalibration.from_json('tests/calib/blender-camera-calib.json'))

marker = Marker(cv2.aruco.DICT_4X4_50, 2, 500)
marker_length = 2.0

segment = Segment(id=1, name='Blender Segment', marker=marker)
segment.add_landmark([[[marker_length, 0.0, 0.0]]])

arucoDetector = ArucoDetector(dictionary=cv2.aruco.DICT_4X4_50, cameraCalibration=blenderCamera.calibration)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok, frame = cap.read()

while ok:
  ok, frame = cap.read()
  _, output = arucoDetector.for_segments(frame, segments=[segment], marker_length=marker_length)
  cv2.imshow('frame', output)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break