import cv2
import numpy as np
import matplotlib.pyplot as plt

"""Define a set of tools for image processing"""

def generate_marker_img(marker_id, marker_size = 500, dictionary = cv2.aruco.DICT_4X4_50):
    """Generate a marker with the given ID"""
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(dictionary), marker_id, marker_size)
    return img

def generate_board(squares: tuple[int, int] = (5,5), square_length: float = 0.04, marker_length: float = 0.02, page_length: int = 640, margin: int = 20, dictionary = cv2.aruco.DICT_4X4_50):
    """Generate a board with the given parameters"""
    board = cv2.aruco.CharucoBoard((squares[0], squares[1]), square_length, marker_length, cv2.aruco.getPredefinedDictionary(dictionary))
    size_ratio = squares[0] / squares[1]
    img = cv2.aruco.CharucoBoard.generateImage(board, (page_length, int(page_length*size_ratio)), marginSize=margin)
    return board, img

def cv2_imshow(cv2image):
    """Takes an cv2 image and displays it (useful for Jupyter Notebooks)"""
    plt.imshow(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
    plt.grid(False)
    plt.axis('off')