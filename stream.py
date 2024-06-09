from ultralytics import YOLO
from PIL import Image
import cv2
import sys


def main():
    model = YOLO("./fire_yolo.pt")
     
    results = model.track(source=0, show=True)


if __name__ == "__main__":
    main()

