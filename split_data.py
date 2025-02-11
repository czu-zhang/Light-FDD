import os, shutil, random
random.seed(0)
import numpy as np
from sklearn.model_selection import train_test_split

val_size = 0.2
test_size = 0.0
postfix = 'jpg'
imgpath = '/home/jovyan/work/datasets/fabricdefect41/images'
txtpath = '/home/jovyan/work/fabricdefect41/labels'

os.makedirs('./dataset/fabricdefect41/train/images', exist_ok=True)
os.makedirs('./dataset/fabricdefect41/valid/images', exist_ok=True)
os.makedirs('./dataset/fabricdefect41/test/images', exist_ok=True)
os.makedirs('./dataset/fabricdefect41/train/labels', exist_ok=True)
os.makedirs('./dataset/fabricdefect41/valid/labels', exist_ok=True)
os.makedirs('./dataset/fabricdefect41/test/labels', exist_ok=True)

listdir = np.array([i for i in os.listdir(txtpath) if 'txt' in i])
random.shuffle(listdir)
train, val, test = listdir[:int(len(listdir) * (1 - val_size - test_size))], listdir[int(len(listdir) * (1 - val_size - test_size)):int(len(listdir) * (1 - test_size))], listdir[int(len(listdir) * (1 - test_size)):]
print(f'train set size:{len(train)} val set size:{len(val)} test set size:{len(test)}')

for i in train:
    #print(i[:-4])
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), './dataset/fabricdefect41/train/images/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), './dataset/fabricdefect41/train/labels/{}'.format(i))

for i in val:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), './dataset/fabricdefect41/valid/images/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), './dataset/fabricdefect41/valid/labels/{}'.format(i))

for i in test:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), './dataset/fabricdefect41/test/images/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), './dataset/fabricdefect41/test/labels/{}'.format(i))