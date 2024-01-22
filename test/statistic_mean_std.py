import os
import glob
import random
import shutil
import numpy as np
from PIL import Image
""" 统计数据库中所有图片的每个通道的均值和标准差 

Totally 3327 files for training
(3327, 128, 128, 3)
[0.47298918 0.43487422 0.32614972]
[0.37761145 0.36143682 0.34962901]
"""

if __name__ == '__main__':

    train_files = glob.glob(os.path.join('train', '*', '*.jpg'))

    print(f'Totally {len(train_files)} files for training')
    result = []
    for file in train_files:
        img = Image.open(file).convert('RGB')
        img = np.array(img).astype(np.uint8) #0~255之间
        img = img/255.  #0~1之间，一般很少用0~255的整形放入网络中，除非做像素点的分类任务
        result.append(img)

    print(np.shape(result)) #[BS,H,W,C]
    mean = np.mean(result, axis=(0,1,2))
    std = np.std(result, axis=(0,1,2))
    print(mean)
    print(std)