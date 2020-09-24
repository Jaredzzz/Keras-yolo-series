from keras.layers import Input, UpSampling2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
import os
import tensorflow as tf
from core.backbone import csp_darknet53_model, _darknet_conv_block, darknet53_model
from core.yolo_layer import YoloLayer


class YOLOV3(object):
    """Implement keras yolov3 here"""

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
            route_1, route_2, input_data = csp_darknet53_model(input_image)
        elif self.backbone == "darknet53":
            route_1, route_2, input_data = darknet53_model(input_image)
        else:
            raise ValueError("Assign correct backbone model: cspdarknet53 or darknet53")

        x = _darknet_conv_block(input_data, convs=[
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_1"},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_2"},
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_3"},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_4"},
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_5"}])
        pred_conv_lbbox = _darknet_conv_block(x, convs=[
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_6"},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'activation': 'linear',
             'layer_idx': "yolo_7"}])

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
        x = _darknet_conv_block(x, convs=[
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_8"}])
        x = UpSampling2D(2)(x)
        x = concatenate([x, route_2])

        x = _darknet_conv_block(x, convs=[
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_11"},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_12"},
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_13"},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_14"},
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_15"}])
        pred_conv_mbbox = _darknet_conv_block(x, convs=[
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_16"},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'activation': 'linear',
             'layer_idx': "yolo_17"}])
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
        x = _darknet_conv_block(x, convs=[
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_18"}])
        x = UpSampling2D(2)(x)
        x = concatenate([x, route_1])

        x = _darknet_conv_block(x, convs=[
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_21"},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_22"},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_23"},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_24"},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_25"}])
        pred_conv_sbbox = _darknet_conv_block(x, convs=[
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolo_26"},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'activation': 'linear',
             'layer_idx': "yolo_27"}])

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

        train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3],
                            [loss_yolo_1, loss_yolo_2, loss_yolo_3])
        infer_model = Model(input_image, [pred_conv_lbbox, pred_conv_mbbox, pred_conv_sbbox])

        return [train_model, infer_model]


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))
