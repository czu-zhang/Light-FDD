#coding:utf-8
from ultralytics import YOLO
# # model inference
model = YOLO('runs/detect/train20/weights/best.pt')
model.predict(source='dataset/test/stain.jpg', **{'save': True})
