#coding:utf-8
from ultralytics import YOLO
# # 模型推理
model = YOLO('runs/detect/train20/weights/best.pt')
model.predict(source='dataset/test/stain.jpg', **{'save': True})