from untitled_core import Marker
import cv2

marker = Marker(cv2.aruco.DICT_6X6_50, 5, 500)
cv2.imshow("Marker", marker.img)

wait_key = cv2.waitKey(0)