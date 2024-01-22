import os
import glob
import random
import shutil
from PIL import Image
""" 对所有图片进行RGB转化，并且统一调整到一致大小，但不让图片发生变形或扭曲，划分了训练集和测试集 """

if __name__ == '__main__':
    test_split_ratio = 0.05 #百分之五的比例作为测试集
    desired_size = 128 # 图片缩放后的统一大小
    raw_path = './raw'

    #把多少个类别算出来，包括目录也包括文件
    dirs = glob.glob(os.path.join(raw_path, '*'))
    #进行过滤，只保留目录，一共36个类别
    dirs = [d for d in dirs if os.path.isdir(d)]

    print(f'Totally {len(dirs)} classes: {dirs}')

    for path in dirs:
        # 对每个类别单独处理

        #只保留类别名称
        path = path.split('/')[-1]

        #创建文件夹
        os.makedirs(f'train/{path}', exist_ok=True)
        os.makedirs(f'test/{path}', exist_ok=True)

        #原始文件夹当前类别的图片进行匹配
        files = glob.glob(os.path.join(raw_path, path, '*.jpg'))
        files += glob.glob(os.path.join(raw_path, path, '*.JPG'))
        files += glob.glob(os.path.join(raw_path, path, '*.png'))

        random.shuffle(files)#原地shuffle，因为要取出来验证集

        boundary = int(len(files)*test_split_ratio) # 训练集和测试集的边界

        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')

            old_size = img.size  # old_size[0] is in (width, height) format

            ratio = float(desired_size)/max(old_size)

            new_size = tuple([int(x*ratio) for x in old_size])#等比例缩放

            im = img.resize(new_size, Image.LANCZOS)#后面的方法不会造成模糊

            new_im = Image.new("RGB", (desired_size, desired_size))

            #new_im在某个尺寸上更大，我们将旧图片贴到上面
            new_im.paste(im, ((desired_size-new_size[0])//2,
                                (desired_size-new_size[1])//2))

            assert new_im.mode == 'RGB'

            if i <= boundary:
                new_im.save(os.path.join(f'test/{path}', file.split('/')[-1].split('.')[0]+'.jpg'))
            else:
                new_im.save(os.path.join(f'train/{path}', file.split('/')[-1].split('.')[0]+'.jpg'))

    test_files = glob.glob(os.path.join('test', '*', '*.jpg'))
    train_files = glob.glob(os.path.join('train', '*', '*.jpg'))

    '''
    Totally 3327 files for training
    Totally 186 files for test
    '''
    print(f'Totally {len(train_files)} files for training')
    print(f'Totally {len(test_files)} files for test')
