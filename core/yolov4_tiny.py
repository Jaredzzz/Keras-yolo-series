from keras.layers import Input, UpSampling2D, Conv2D, BatchNormalization, Lambda, MaxPooling2D, LeakyReLU, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model
import tensorflow as tf
from core.yolo_layer import YoloLayer
from core.activation import Mish, Mish6
from core.backbone import yolov4_tiny_backbone, _darknet_conv_block


class YOLOV4_tiny(object):
    """Implement keras YOLOV4_tiny here"""

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
        true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4))  # xywh
        true_yolo_1 = Input(
            shape=(None, None, len(self.anchors) // 4, 4 + 1 + self.num_class))  # grid_h, grid_w, nb_anchor, 4+1+nb_class
        true_yolo_2 = Input(
            shape=(None, None, len(self.anchors) // 4, 4 + 1 + self.num_class))  # grid_h, grid_w, nb_anchor, 4+1+nb_class

        if self.backbone == "YOLOV4_tiny_backbone":
            print("[INFO] Backbone: YOLOV4_tiny_backbone ")
            route, input_data = yolov4_tiny_backbone(input_image)
        else:
            raise ValueError("Assign correct backbone model: YOLOV4_tiny_backbone")

        x = _darknet_conv_block(input_data, convs=[
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolov4_tiny_1"},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolov4_tiny_2"}])
        pred_conv_lbbox = _darknet_conv_block(x, convs=[
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolov4_tiny_3"},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'activation': 'linear',
             'layer_idx': "yolov4_tiny_4"}])

        loss_yolo_1 = YoloLayer(self.anchors[6:],
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
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolov4_tiny_5"}])
        x = UpSampling2D(2)(x)
        x = concatenate([x, route])

        pred_conv_sbbox = _darknet_conv_block(x, convs=[
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': "yolov4_tiny_6"},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'activation': 'linear',
             'layer_idx': "yolo_5"}])
        loss_yolo_2 = YoloLayer(self.anchors[:6],
                                [1 * num for num in self.max_grid],
                                self.batch_size,
                                self.warmup_batches,
                                self.iou_loss_thresh,
                                self.grid_scales[1],
                                self.obj_scale,
                                self.noobj_scale,
                                self.xywh_scale,
                                self.class_scale,
                                self.iou_loss,
                                self.focal_loss)([input_image, pred_conv_sbbox, true_yolo_2, true_boxes])

        train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2], [loss_yolo_1, loss_yolo_2])
        infer_model = Model(input_image, [pred_conv_lbbox, pred_conv_sbbox])
        train_model.summary()
        return [train_model, infer_model]


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))


