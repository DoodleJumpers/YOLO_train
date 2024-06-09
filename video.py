from ultralytics import YOLO
from PIL import Image
import cv2
import sys


def main():
    model = YOLO("./best_2.pt")
     
    results = model.track(source="./Лесной пожар под Тверью сняли на видео с квадрокоптера.mp4", show=True, save=True, project='./result')


if __name__ == "__main__":
    main()

