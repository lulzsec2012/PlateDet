#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from config import cfg

class plate_det:
    '''
    convert final layer features to bounding box parameters.
    '''
    def __init__(self, anchors, num_classes, img_shape):
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_shape = img_shape

    def build(self, feats):
        anchors_tensor = tf.reshape(self.anchors, [1,1,1,cfg.num_anchors_per_layer,2])

        conv_dims = tf.stack([tf.shape(feats)[2], tf.shape(feats)[1]])
        conv_height_index = tf.range(conv_dims[1])
        conv_width_index = tf.range(conv_dims[0])
        conv_width_index, conv_height_index = tf.meshgrid(conv_width_index, conv_height_index)
        conv_height_index = tf.reshape(conv_height_index, [-1, 1])
        conv_width_index = tf.reshape(conv_width_index, [-1, 1])
        conv_index = tf.concat([conv_width_index, conv_height_index], axis=-1)

        conv_index = tf.reshape(conv_index, [1, conv_dims[1], conv_dims[0], 1, 2])
        conv_index = tf.cast(conv_index, tf.float32)

        feats = tf.reshape(feats, [-1, conv_dims[1], conv_dims[0], cfg.num_anchors_per_layer, self.num_classes + 5])

        conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), tf.float32)
        img_dims = tf.stack([self.img_shape[2], self.img_shape[1]])
        img_dims = tf.cast(tf.reshape(img_dims, [1, 1, 1, 1, 2]), tf.float32)

        box_xy = tf.sigmoid(feats[..., :2])
        box_twh = feats[..., 2:4]
        box_wh = tf.exp(box_twh)
        self.box_confidence = tf.sigmoid(feats[..., 4:5])
        self.box_class_probs = tf.sigmoid(feats[..., 5:])
        self.box_xy = (box_xy + conv_index) / conv_dims
        self.box_wh = box_wh * anchors_tensor / img_dims
        self.loc_txywh = tf.concat([box_xy, box_twh], axis=-1)

        return self.box_xy, self.box_wh, self.box_confidence, self.box_class_probs, self.loc_txywh

def preprocess_true_boxes(true_boxes, anchors, feat_size, image_size):
    num_anchors = cfg.num_anchors_per_layer

    true_wh = tf.expand_dims(true_boxes[..., 2:4], 2)
    true_wh_half = true_wh / 2.
    true_mins = 0 - true_wh_half
    true_maxes = true_wh_half

    img_wh = tf.reshape(tf.stack([image_size[2], image_size[1]]), [1, -1])
    anchors = anchors / tf.cast(img_wh, tf.float32)
    anchors_shape = tf.shape(anchors)
    anchors = tf.reshape(anchors, [1, 1, anchors_shape[0], anchors_shape[1]])
    anchors_half = anchors / 2.
    anchors_mins = 0 - anchors_half
    anchors_maxes = anchors_half

    intersect_mins = tf.maximum(true_mins, anchors_mins)
    intersect_maxes = tf.minimum(true_maxes, anchors_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    anchors_areas = anchors[..., 0] * anchors[..., 1]

    union_areas = true_areas + anchors_areas - intersect_areas

    iou_scores = intersect_areas / union_areas
    valid = tf.logical_not(tf.reduce_all(tf.equal(iou_scores, 0), axis=-1))
    iout_argmax = tf.cast(tf.argmax(iou_scores, axis=-1), tf.int32)
    anchors = tf.reshape(anchors, [-1, 2])
    anchors_cf = tf.gather(anchors, iout_argmax)

    feat_wh = tf.reshape(tf.stack([feat_size[2], feat_size[1]]), [1, -1])
    cxy = tf.cast(tf.floor(true_boxes[..., :2] * tf.cast(feat_wh, tf.float32)), tf.int32)
    sig_xy = tf.cast(true_boxes[..., :2] * tf.cast(feat_wh, tf.float32) - tf.cast(cxy, tf.float32), tf.float32)
    idx = cxy[..., 1] * (num_anchors * feat_size[2]) + num_anchors * cxy[..., 0] + iout_argmax
    idx_one_hot = tf.one_hot(idx, depth=feat_size[1] * feat_size[2] * num_anchors)
    idx_one_hot = tf.reshape(idx_one_hot, [-1, cfg.train.max_truth, feat_size[1], feat_size[2], num_anchors, 1])
    loc_scale = 2 - true_boxes[..., 2] * true_boxes[..., 3]
    mask = []
    loc_cls = []
    scale = []
    for i in range(cfg.batch_size):
        idx_i = tf.where(valid[i])[:, 0]
        mask_i = tf.gather(idx_one_hot[i], idx_i)

        scale_i = tf.gather(loc_scale[i], idx_i)
        scale_i = tf.reshape(scale_i, [-1, 1, 1, 1, 1])
        scale_i = scale_i * mask_i
        scale_i = tf.reduce_sum(scale_i, axis=0)
        scale_i = tf.maximum(tf.minimum(scale_i, 2), 1)
        scale.append(scale_i)

        true_boxes_i = tf.gather(true_boxes[i], idx_i)
        sig_xy_i = tf.gather(sig_xy[i], idx_i)
        anchors_cf_i = tf.gather(anchors_cf[i], idx_i)
        twh = tf.log(true_boxes_i[:, 2:4] / anchors_cf_i)
        loc_cls_i = tf.concat([sig_xy_i, twh, true_boxes_i[:, -1:]], axis=-1)
        loc_cls_i = tf.reshape(loc_cls_i, [-1, 1, 1, 1, 5])
        loc_cls_i = loc_cls_i * mask_i
        loc_cls_i = tf.reduce_sum(loc_cls_i, axis=[0])

        loc_cls_i = tf.concat([loc_cls_i[..., :4], tf.minimum(loc_cls_i[..., -1:], 19)], axis=-1)
        loc_cls.append(loc_cls_i)

        mask_i = tf.reduce_sum(mask_i, axis=[0])
        mask_i = tf.minimum(mask_i, 1)
        mask.append(mask_i)

    loc_cls = tf.stack(loc_cls, axis=0)
    mask = tf.stack(mask, axis=0)
    scale = tf.stack(scale, axis=0)
    return loc_cls, mask, scale


def confidence_loss(pred_xy, pred_wh, pred_confidence, true_boxes, detectors_mask):
    pred_xy = tf.expand_dims(pred_xy, 4)
    pred_wh = tf.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = tf.shape(true_boxes)
    true_boxes = tf.reshape(true_boxes, [true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    best_ious = tf.reduce_max(iou_scores, axis=-1, keepdims=True)

    object_ignore = tf.cast(best_ious > cfg.train.ignore_thresh, best_ious.dtype)
    no_object_weights = (1 - object_ignore) * (1 - detectors_mask)
    no_objects_loss = no_object_weights * tf.square(pred_confidence)
    objects_loss = detectors_mask * tf.square(1 - pred_confidence)

    objectness_loss = tf.reduce_sum(objects_loss + no_objects_loss)
    #objectness_loss = tf.reduce_mean(objects_loss + no_objects_loss)
    return objectness_loss


def cord_cls_loss(
                detectors_mask,
                matching_true_boxes,
                num_classes,
                pred_class_prob,
                pred_boxes,
                loc_scale,
              ):
    matching_classes = tf.cast(matching_true_boxes[..., 4], tf.int32)
    matching_classes = tf.one_hot(matching_classes, num_classes)
    classification_loss = (detectors_mask * tf.square(matching_classes - pred_class_prob))

    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (detectors_mask * loc_scale * tf.square(matching_boxes - pred_boxes))

    classification_loss_sum = tf.reduce_sum(classification_loss)
    #classification_loss_sum = tf.reduce_mean(classification_loss)
    coordinates_loss_sum = tf.reduce_sum(coordinates_loss)
    #coordinates_loss_sum = tf.reduce_mean(coordinates_loss)

    return classification_loss_sum + coordinates_loss_sum

