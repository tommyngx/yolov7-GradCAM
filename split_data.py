import os
import shutil

sets = ['train', 'val']
file_suffix = ['.jpg', '.txt']
root_path = 'datasets/Fruit/'
txt_path = 'datasets/Fruit/ImageSets/'

# 开始遍历
for s in sets:
    for fs in file_suffix:
        read_and_save_path = root_path + 'images/' if fs == '.jpg' else root_path + 'labels/'
        if not os.path.exists(read_and_save_path + s):
            os.mkdir(read_and_save_path + s)
        num = 0
        with open(txt_path + s + '.txt', 'r') as f:
            for name in f:
                fileName = name.strip()

                for file in os.listdir(read_and_save_path):
                    if fileName + fs == file:
                        num += 1
                        shutil.move(os.path.join(read_and_save_path, fileName + fs), read_and_save_path + s)
            print("Copy complete!")
            print("Total pictures copied:", num)
