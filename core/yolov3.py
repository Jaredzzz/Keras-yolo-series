from keras.layers import Dense,Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D,UpSampling2D
from keras.layers.merge import add,concatenate
from keras.models import Model
from keras.initializers import glorot_uniform
import os
import tensorflow as tf
from keras.engine.topology import Layer
import numpy as np
from core.backbone import csp_darknet53_model,_darknet_conv_block


class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh,
                 grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale,
                 **kwargs):
        # make the model settings persistent
        self.ignore_thresh = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors = tf.constant(anchors, dtype='float', shape=[1, 1, 1, 3, 2])
        self.grid_scale = grid_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.xywh_scale = xywh_scale
        self.class_scale = class_scale

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # y_pred: xywh+conf+class_conf
        # y_true: xywh+conf+class_conf
        # true_boxes: xywh
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

        # initialize the masks
        # 真实数据置信度
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)

        # compute grid factor and net factor
        #
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_h = tf.shape(input_image)[1]
        net_w = tf.shape(input_image)[2]
        net_factor = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

        """
        Adjust prediction
        """
        pred_box_xy = (self.cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh = y_pred[..., 2:4]  # t_wh
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)  # adjust confidence
        pred_box_class = y_pred[..., 5:]  # adjust class probabilities

        """
        Adjust ground truth
        """
        true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
        true_box_wh = y_true[..., 2:4]  # t_wh
        true_box_conf = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Compare each predicted box to all true boxes
        """
        # initially, drag all objectness of all boxes to 0
        conf_delta = pred_box_conf - 0

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        # 找到左上角最大的的坐标和右下角最小的坐标
        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        # 计算并集，并计算IOU
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)   # 返回以浮点数进行计算的 intersect_areas/union_areas
        # 找出与真实框IOU最大的边界框
        best_ious = tf.reduce_max(iou_scores, axis=4)
        # 如果最大的IOU小于阈值, 那么认为不包含目标,则为背景框
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

        """
        Compute some online statistics
        """
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)
        iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

        count = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.to_float((pred_box_conf * object_mask) >= 0.5)
        class_mask = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50 = tf.reduce_sum(tf.to_float(iou_scores >= 0.5) * detect_mask * class_mask) / (count + 1e-3)
        recall75 = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask * class_mask) / (count + 1e-3)
        avg_iou = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj = tf.reduce_sum(pred_box_conf * object_mask) / (count + 1e-3)
        avg_noobj = tf.reduce_sum(pred_box_conf * (1 - object_mask)) / (count_noobj + 1e-3)
        avg_cat = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)

        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches + 1),
                                                      lambda: [true_box_xy + (
                                                                  0.5 + self.cell_grid[:, :grid_h, :grid_w, :, :]) * (
                                                                           1 - object_mask),
                                                               true_box_wh + tf.zeros_like(true_box_wh) * (
                                                                           1 - object_mask),
                                                               tf.ones_like(object_mask)],
                                                      lambda: [true_box_xy,
                                                               true_box_wh,
                                                               object_mask])

        """
        Compare each true box to all anchor boxes
        """
        # 差平方和误差（sum-squared error）
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        box_loss_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1],
                                  axis=4)  # the smaller the box, the bigger the scale

        xy_delta = xywh_mask * (pred_box_xy - true_box_xy) * box_loss_scale * self.xywh_scale
        wh_delta = xywh_mask * (pred_box_wh - true_box_wh) * box_loss_scale * self.xywh_scale
        # 计算置信度损失，原理是利用最大iou如果大于阈值才认为目标框含有检测目标
        conf_delta = object_mask * (pred_box_conf - true_box_conf) * self.obj_scale + (
                    1 - object_mask) * conf_delta * self.noobj_scale
        # 分类交叉熵损失
        class_delta = object_mask * \
                      tf.expand_dims(
                          tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class),
                          4) * \
                      self.class_scale

        loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
        loss_wh = tf.reduce_sum(tf.square(wh_delta), list(range(1, 5)))
        loss_conf = tf.reduce_sum(tf.square(conf_delta), list(range(1, 5)))
        loss_class = tf.reduce_sum(class_delta, list(range(1, 5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy),
                               tf.reduce_sum(loss_wh),
                               tf.reduce_sum(loss_conf),
                               tf.reduce_sum(loss_class)], message='loss xy, wh, conf, class: \t', summarize=1000)

        return loss * self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]

    def focal(self, target, actual, alpha=1, gamma=2):
        # 目标检测中, 通常正样本较少, alpha可以调节正负样本的比例,
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):
        # boxes:[x,y,w,h]转化为[xmin,ymin,xmax,ymax]
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        # 计算boxe1和boxes2的面积
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # 计算boxe1和boxes2交集的左上角和右下角坐标
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        # 计算交集区域的宽高, 没有交集,宽高为置0
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        # 计算最小闭合面C的左上角和右下角坐标
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])

        # 计算最小闭合面C的宽高
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]

        # 计算GIOU
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):
        # 分别计算2个边界框的面积
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        # (x,y,w,h)->(xmin,ymin,xmax,ymax)
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        # 找到左上角最大的的坐标和右下角最小的坐标
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
        # 判断右下角的坐标是不是大于左上角的坐标，大于则有交集，否则没有交集
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        # 计算并集，并计算IOU
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou


class YOLOV3(object):
    """Implement keras yolov3 here"""

    def __init__(self, config,max_box_per_image,batch_size,warmup_batches):

        self.classes = config["model"]["labels"]
        self.num_class = len(self.classes)
        self.anchors = config["model"]["anchors"]
        self.grid_scales = config["train"]["grid_scales"]
        self.obj_scale = config["train"]["obj_scale"]
        self.noobj_scale = config["train"]["noobj_scale"]
        self.xywh_scale = config["train"]["xywh_scale"]
        self.class_scale = config["train"]["class_scale"]
        self.iou_loss_thresh = config["train"]["iou_loss_thresh"]
        self.max_grid = [config['model']['max_input_size'], config['model']['max_input_size']]
        self.batch_size = batch_size
        self.warmup_batches = warmup_batches
        self.max_box_per_image = max_box_per_image


    def model(self):
        input_image = Input(shape=(None, None, 3))  # net_h, net_w, 3
        true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4))
        true_yolo_1 = Input(
            shape=(None, None, len(self.anchors) // 6, 4 + 1 + self.num_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
        true_yolo_2 = Input(
            shape=(None, None, len(self.anchors) // 6, 4 + 1 + self.num_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
        true_yolo_3 = Input(
            shape=(None, None, len(self.anchors) // 6, 4 + 1 + self.num_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
        route_1, route_2, input_data = csp_darknet53_model(input_image)

        x = _darknet_conv_block(input_data,convs=[
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_1"},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_2"},
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_3"},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_4"},
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_5"}])
        pred_conv_lbbox = _darknet_conv_block(x,convs=[
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_6"},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': "yolo_7"}])

        loss_yolo_1 = YoloLayer(self.anchors[12:],
                                [1 * num for num in self.max_grid],
                                self.batch_size,
                                self.warmup_batches,
                                self.iou_loss_thresh,
                                self.grid_scales[0],
                                self.obj_scale,
                                self.noobj_scale,
                                self.xywh_scale,
                                self.class_scale)([input_image, pred_conv_lbbox, true_yolo_1, true_boxes])
        x = _darknet_conv_block(x, convs=[
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_8"}])
        x = UpSampling2D(2)(x)
        x = concatenate([x, route_2])

        x = _darknet_conv_block(x,convs=[
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_11"},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_12"},
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_13"},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_14"},
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': "yolo_15"}])
        pred_conv_mbbox = _darknet_conv_block(x,convs=[
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_16"},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': "yolo_17"}])
        loss_yolo_2 = YoloLayer(self.anchors[6:12],
                                [2 * num for num in self.max_grid],
                                self.batch_size,
                                self.warmup_batches,
                                self.iou_loss_thresh,
                                self.grid_scales[1],
                                self.obj_scale,
                                self.noobj_scale,
                                self.xywh_scale,
                                self.class_scale)([input_image, pred_conv_mbbox, true_yolo_2, true_boxes])
        x = _darknet_conv_block(x, convs=[
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_18"}])
        x = UpSampling2D(2)(x)
        x = concatenate([x, route_1])

        x = _darknet_conv_block(x, convs=[
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_21"},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_22"},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_23"},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_24"},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_25"}])
        pred_conv_sbbox = _darknet_conv_block(x,convs=[
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': "yolo_26"},
            {'filter': (3 * (5 + self.num_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': "yolo_27"}])

        loss_yolo_3 = YoloLayer(self.anchors[:6],
                                [4 * num for num in self.max_grid],
                                self.batch_size,
                                self.warmup_batches,
                                self.iou_loss_thresh,
                                self.grid_scales[2],
                                self.obj_scale,
                                self.noobj_scale,
                                self.xywh_scale,
                                self.class_scale)([input_image, pred_conv_sbbox, true_yolo_3, true_boxes])

        train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3],
                            [loss_yolo_1, loss_yolo_2, loss_yolo_3])
        infer_model = Model(input_image, [pred_conv_lbbox, pred_conv_mbbox, pred_conv_sbbox])

        return [train_model, infer_model]



def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))
