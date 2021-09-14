"""
Classify a few images through our CNN.
"""
import numpy as np
from utils.utils import preprocess_input
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
import cv2
from skimage import io
from core.activation import Mish, Mish6
import tensorflow as tf
import os
from keras.utils import plot_model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    model = load_model('../backup/RDNet_giou_100.h5',
                       compile=False,
                       custom_objects={'Mish': Mish, 'Mish6': Mish6, 'tf': tf})
    # replaced by your model name
    plot_model(model, "infer_RDNet.png", show_shapes=True)
    # Get all our test images.
    image_name = '/data/zjj_workspace/RDNet_data/dataset/images/valid/peeling_1000mm_5_Intensity_8.tiff'
    image = cv2.imread(image_name)
    # cv2.imshow("Image", intensity_image)
    # cv2.waitKey(0)
    # Turn the image into an array.
    image_arr = preprocess_input(image, net_h=512, net_w=512)  # 根据载入的训练好的模型的配置，将图像统一尺寸

    # 设置可视化的层
    name_list = []
    for i in range(130):
        name_list.append(model.layers[i].name)
    output_layer = name_list.index('concatenate_3')
    print(output_layer)
    layer_1 = K.function([model.layers[0].input], [model.layers[output_layer].output])
    f1 = layer_1([image_arr])[0]
    for _ in range(16):
        show_img = f1[:, :, :, _]
        show_img.shape = [16, 16]
        plt.subplot(4, 4, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()
    # # conv layer: 299
    # layer_1 = K.function([model.layers[0].input], [model.layers[299].output])
    # f1 = layer_1([image_arr])[0]
    # for _ in range(81):
    #     show_img = f1[:, :, :, _]
    #     show_img.shape = [8, 8]
    #     plt.subplot(9, 9, _ + 1)
    #     plt.imshow(show_img, cmap='gray')
    #     plt.axis('off')
    # plt.show()
    print('This is the end !')


if __name__ == '__main__':
    main()
