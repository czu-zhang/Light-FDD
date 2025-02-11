#coding:utf-8
from ultralytics import YOLO
# # 模型推理
model = YOLO('runs/detect/train120/weights/best.pt')
model.predict(source='datasets/test/stain.jpg', **{'save': True})