import numpy as np


def filter_boxes(box_xywh, scores, score_threshold: float, input_shape):
    scores_max = np.amax(scores, axis=-1)
    mask = scores_max >= score_threshold

    class_boxes = box_xywh[0][tuple(mask)]
    pred_conf = scores[0][tuple(mask)]

    class_boxes = np.reshape(class_boxes, [np.shape(scores)[0], -1, np.shape(class_boxes)[-1]])
    pred_conf = np.reshape(pred_conf, [np.shape(scores)[0], -1, np.shape(pred_conf)[-1]])

    box_xy, box_wh = np.split(class_boxes, indices_or_sections=[2], axis=-1)

    input_shape = np.array(input_shape, dtype=np.float32)

    box_mins = (box_xy - (box_wh / 2.)) / input_shape
    box_maxes = (box_xy + (box_wh / 2.)) / input_shape

    boxes = np.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2]  # y_max
    ], axis=-1)

    return boxes, pred_conf


def non_max_suppression(boxes, max_bbox_overlap, scores, frame_size):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[0, :, 0] * frame_size[0]
    y1 = boxes[0, :, 1] * frame_size[1]
    x2 = boxes[0, :, 2] * frame_size[0]
    y2 = boxes[0, :, 3] * frame_size[1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick
