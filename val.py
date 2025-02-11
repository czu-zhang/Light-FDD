#coding:utf-8
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('./runs/detect/train/weights/last.pt')
    model.val( data='./dataset/fabricdefect41/data.yaml',
               split='val',
               imgsz=640,
               batch=16,
               project='runs/val',
               name='exp',
               )
    # model.train(data='/home/datasets/TomatoData/data.yaml', epochs=50, batch=4)
