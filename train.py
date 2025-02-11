#coding:utf-8
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-FasterNet_ContextGuided_RFCBAM_one_vovd.yaml')
    model.train(
        **{'cfg': 'ultralytics/cfg/exp1.yaml', 'data': 'dataset/fabricdefect41/data.yaml', 'epochs': 300, 'amp': False})
