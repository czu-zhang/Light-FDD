#coding:utf-8
from ultralytics import YOLO
# # model inference
model = YOLO('runs/detect/train120/weights/best.pt')
model.predict(source='datasets/test/stain.jpg', **{'save': True})
