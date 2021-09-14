#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from scripts.voc import parse_voc_annotation
from scripts.generator import BatchGenerator
from utils.utils import normalize, evaluate
from keras.models import load_model
from core.activation import Mish, Mish6
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Create the validation generator
    ###############################  
    test_ints, labels = parse_voc_annotation(
        config['test']['test_annot_folder'],
        config['test']['test_image_folder'],
        config['test']['cache_name'],
        config['model']['labels']
    )

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)
   
    test_generator = BatchGenerator(
        instances           = test_ints,
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize,
        tiny                = config['model']['tiny']
    )

    ###############################
    #   Load the model and do evaluation
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    print("[INFO] Start Evalutating Model...")
    print("[INFO] Loading weights: %s " % config['train']['saved_weights_name'])
    infer_model = load_model(config['train']['saved_weights_name'],
                             compile=False,
                             custom_objects={'Mish': Mish, 'Mish6': Mish6, 'tf': tf})

    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, test_generator,
                                  iou_threshold=config['test']['iou_threshold'],
                                  obj_thresh=config['test']['obj_thresh'],
                                  tiny=config['model']['tiny'])
    ap = []

    # print the mAP score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ' average precision(AP): {:.6f}'.format(average_precision['ap']))
        ap.append(average_precision['ap'])
        print(labels[label] + ' recall: {:.6f}'.format(average_precision['recall']))
        print(labels[label] + ' precision: {:.6f}'.format(average_precision['precision']))
    print('[INFO] mAP: {:.6f}'.format(sum(ap) / len(ap)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate keras YOLOV4_tiny or YOLOV4 on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')    
    
    args = argparser.parse_args()
    _main_(args)
