from untitled_core import Marker, Camera, ArucoDetector, CameraCalibration, Pose
import cv2

marker = Marker(cv2.aruco.DICT_4X4_50, 2, 500)
cv2.imshow("Marker", marker.img)

wait_key = cv2.waitKey(0)