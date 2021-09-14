import os
import numpy as np
import random
import shutil

val_percent = 0.1
train_percent = 0.8
test_percent = 0.1
image_path = "E:/Data/underwater/train/image"
xml_path = "E:/Data/underwater/train/box"

image_savepath = "E:/Data/underwater/dataset/images"
annotation_savepath = "E:/Data/underwater/dataset/annotations"

if os.path.exists(image_savepath):
    pass
else:
    os.mkdir(image_savepath)

if os.path.exists(annotation_savepath):
    pass
else:
    os.mkdir(annotation_savepath)

for type in ["train", "test", "valid"]:
    if os.path.exists(os.path.join(image_savepath, type)):
        for root, dirs, files in os.walk(os.path.join(image_savepath, type), topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.mkdir(os.path.join(image_savepath, type))

    if os.path.exists(os.path.join(annotation_savepath, type)):
        for root, dirs, files in os.walk(os.path.join(annotation_savepath, type), topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.mkdir(os.path.join(annotation_savepath, type))

total_image = os.listdir(image_path)

num = len(total_image)
print("all_num: %s" % num)
# list = range(num)
random.shuffle(total_image)
valid = int(num * val_percent)  # val
test = int(num * test_percent)  # test
count = 0

for image in total_image:
    print(image)
    image_file_path = os.path.join(image_path, image)
    xml_file_path = os.path.join(xml_path, image.replace('jpg', 'xml'))
    if count < valid:
        shutil.copy(image_file_path, os.path.join(image_savepath, "valid"))
        shutil.copy(xml_file_path, os.path.join(annotation_savepath, "valid"))
    elif valid <= count < valid+test:
        shutil.copy(image_file_path, os.path.join(image_savepath, "test"))
        shutil.copy(xml_file_path, os.path.join(annotation_savepath, "test"))
    else:
        shutil.copy(image_file_path, os.path.join(image_savepath, "train"))
        shutil.copy(xml_file_path, os.path.join(annotation_savepath, "train"))
    count += 1
