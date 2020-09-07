# -*- coding: utf-8 -*-
import keras
import numpy as np
import matplotlib.pyplot as plt
import time

class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.loss = {'batch': [], 'epoch': []}
        self.loss_1 = {'batch': [], 'epoch': []}
        self.loss_2 = {'batch': [], 'epoch': []}
        self.loss_3 = {'batch': [], 'epoch': []}

    # 按照batch来进行追加数据
    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.loss['batch'].append(logs.get('yolo_layer_1_loss') + logs.get('yolo_layer_2_loss')
                                  + logs.get('yolo_layer_3_loss'))
        self.loss_1['batch'].append(logs.get('yolo_layer_1_loss'))
        self.loss_2['batch'].append(logs.get('yolo_layer_2_loss'))
        self.loss_3['batch'].append(logs.get('yolo_layer_3_loss'))
        # 每10秒按照当前容器里的值来绘图
        if int(time.time()) % 10 == 0:
            self.draw_p([self.loss['batch'],self.loss_1['batch'],self.loss_2['batch'],self.loss_3['batch']],
                        ['loss','yolo_layer_1_loss','yolo_layer_2_loss','yolo_layer_3_loss'],
                        'train_batch')

    # def on_epoch_end(self, batch, logs={}):
    #     self.loss['epoch'].append(logs.get('yolo_layer_1_loss') + logs.get('yolo_layer_2_loss')
    #                               + logs.get('yolo_layer_3_loss'))
    #     # 每10秒按照当前容器里的值来绘图
    #     if int(time.time()) % 10 == 0:
    #         self.draw_p(self.loss['epoch'], 'loss', 'train_epoch')

    # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type):
        plt.figure()
        for i in range(len(lists)):
            plt.plot(range(len(lists[i])), lists[i], label=label[i])
        plt.ylabel("Loss")
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label[0] + '.png')

    # 由于这里的绘图设置的是10s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-10秒的时间）
    # 所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        self.draw_p(self.loss['batch'], 'loss', 'train_batch')