import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import cv2
from src.untitled_aruco_package.untitled_core import Camera, CameraCalibration
from src.untitled_aruco_package.untitled_utils import generate_board

blenderCamera = Camera(id=0, reference='Blender Camera')

calibrationBoard, calibrationBoardImg = generate_board()
calibImgs = []
for i in range(1, 5):
  calibImgs.append(cv2.imread(f'tests/img/blender-camera-calib/camera-calib-01.png'))

if calibImgs.__len__() == 0:
  raise Exception("No calibration images found")

blenderCameraCalibration, _ = blenderCamera.calibrate(calibImgs, calibrationBoard)

# save the calibration
blenderCameraCalibration.save('tests/calib/blender-camera-calib.json')

print(f'Camera matrix: \n{blenderCameraCalibration.camera_matrix} \n')
print(f'Distortion Coeffs: \n{blenderCameraCalibration.dist_coeffs}')
