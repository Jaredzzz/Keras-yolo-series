from keras.layers import Input, UpSampling2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
import os
import tensorflow as tf
from core.backbone import csp_darknet53_model, _darknet_conv_block, darknet53_model
from core.yolo_layer import YoloLayer


class YOLOV4(object):
    """Implement keras yolov4 here"""

    def __init__(self, config, max_box_per_image, batch_size, warmup_batches):

        self.classes = config["model"]["labels"]
        self.num_class = len(self.classes)
        self.anchors = config["model"]["anchors"]
        self.grid_scales = config["train"]["grid_scales"]
        self.obj_scale = config["train"]["obj_scale"]
        self.noobj_scale = config["train"]["noobj_scale"]
        self.xywh_scale = config["train"]["xywh_scale"]
        self.class_scale = config["train"]["class_scale"]
        self.iou_loss_thresh = config["train"]["iou_loss_thresh"]
        self.iou_loss = config["train"]["iou_loss"]
        self.max_grid = [config['model']['max_input_size'], config['model']['max_input_size']]
        self.batch_size = batch_size
        self.warmup_batches = warmup_batches
        self.max_box_per_image = max_box_per_image
        self.focal_loss = config["train"]["focal_loss"]
        self.backbone = config["model"]["backbone_model"]

    def model(self):
        input_image = Input(shape=(None, None, 3))  # net_h, net_w, 3
        true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4))
        true_yolo_1 = Input(
            shape=(None, None, len(self.anchors) // 6, 4 + 1 + self.num_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
        true_yolo_2 = Input(
            shape=(None, None, len(self.anchors) // 6, 4 + 1 + self.num_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
        true_yolo_3 = Input(
            shape=(None, None, len(self.anchors) // 6, 4 + 1 + self.num_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
        if self.backbone == "cspdarknet53":
            route_1, route_2, input_data = csp_darknet53_model(input_image)  # 8,16,32倍
        elif self.backbone == "darknet53":
            route_1, route_2, input_data = darknet53_model(input_image)  # 8,16,32倍
        else:
            raise ValueError("Assign correct backbone model: cspdarknet53 or darknet53")
        # SPP
        x = yolo_spp(input=input_data)
        # PANet Neck
        pan_sbbox, pan_mbbox, pan_lbbox = yolo_panet(route_1, route_2, x)  # 8,16,32倍

        # yolo head
        # 8倍降采样:76x76
        pred_conv_sbbox = _darknet_conv_block(pan_sbbox, convs=[
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 137},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'activation': 'linear',
             'layer_idx': 138}])

        loss_yolo_3 = YoloLayer(self.anchors[:6],
                                [4 * num for num in self.max_grid],
                                self.batch_size,
                                self.warmup_batches,
                                self.iou_loss_thresh,
                                self.grid_scales[2],
                                self.obj_scale,
                                self.noobj_scale,
                                self.xywh_scale,
                                self.class_scale,
                                self.iou_loss,
                                self.focal_loss)([input_image, pred_conv_sbbox, true_yolo_3, true_boxes])
        # 16倍降采样:38x38
        pred_conv_mbbox = _darknet_conv_block(pan_mbbox, convs=[
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 148},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'activation': 'linear',
             'layer_idx': 149}])

        loss_yolo_2 = YoloLayer(self.anchors[6:12],
                                [2 * num for num in self.max_grid],
                                self.batch_size,
                                self.warmup_batches,
                                self.iou_loss_thresh,
                                self.grid_scales[1],
                                self.obj_scale,
                                self.noobj_scale,
                                self.xywh_scale,
                                self.class_scale,
                                self.iou_loss,
                                self.focal_loss)([input_image, pred_conv_mbbox, true_yolo_2, true_boxes])
        # 32倍降采样:19x19
        pred_conv_lbbox = _darknet_conv_block(pan_lbbox, convs=[
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 159},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'activation': 'linear',
             'layer_idx': 160}])

        loss_yolo_1 = YoloLayer(self.anchors[12:],
                                [1 * num for num in self.max_grid],
                                self.batch_size,
                                self.warmup_batches,
                                self.iou_loss_thresh,
                                self.grid_scales[0],
                                self.obj_scale,
                                self.noobj_scale,
                                self.xywh_scale,
                                self.class_scale,
                                self.iou_loss,
                                self.focal_loss)([input_image, pred_conv_lbbox, true_yolo_1, true_boxes])

        train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3],
                            [loss_yolo_1, loss_yolo_2, loss_yolo_3])
        infer_model = Model(input_image, [pred_conv_lbbox, pred_conv_mbbox, pred_conv_sbbox])

        return [train_model, infer_model]


def yolo_spp(input):

    x = _darknet_conv_block(input, convs=[
        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_spp_1"},
        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_spp_2"},
        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_spp_3"}])

    x1 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)
    x2 = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)
    x3 = MaxPooling2D(pool_size=(13, 13), strides=1, padding='same')(x)

    x = concatenate([x, x1, x2, x3])

    return x


def yolo_panet(route_1, route_2, input):

    x = _darknet_conv_block(input, convs=[
        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_1"},
        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_2"},
        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_3"}])

    pan_lbbox = x

    x = _darknet_conv_block(x, convs=[
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_4"}])
    x = UpSampling2D(2)(x)
    route_2 = _darknet_conv_block(route_2, convs=[
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_6"}])
    x = concatenate([x, route_2])

    x = _darknet_conv_block(x, convs=[
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_8"},
        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_9"},
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_10"},
        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_11"},
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_12"}])

    pan_mbbox = x

    x = _darknet_conv_block(x, convs=[
        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_13"}])
    x = UpSampling2D(2)(x)
    route_1 = _darknet_conv_block(route_1, convs=[
        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_15"}])
    pan_sbbox = concatenate([x, route_1])

    # 8倍降采样:76x76
    x = _darknet_conv_block(pan_sbbox, convs=[
        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_17"},
        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_18"},
        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_19"},
        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_20"},
        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_21"}])

    pan_sbbox = x

    # 16倍降采样:38x38
    x = _darknet_conv_block(x, convs=[
        {'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_22"}])
    x = concatenate([x,pan_mbbox])

    x = _darknet_conv_block(x, convs=[
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_24"},
        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_25"},
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_26"},
        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_27"},
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_28"}])
    pan_mbbox = x
    # 32倍降采样:19x19
    x = _darknet_conv_block(x, convs=[
        {'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_29"}])
    x = concatenate([x,pan_lbbox])

    x = _darknet_conv_block(x, convs=[
        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_31"},
        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_32"},
        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_33"},
        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_34"},
        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_pan_35"}])

    pan_lbbox = x

    return pan_sbbox, pan_mbbox, pan_lbbox
