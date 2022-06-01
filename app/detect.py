import logging
import time

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 0)  # black
_BOX_COLOR = (0, 255, 0)  # green

input_size = 416
iou = 0.45
score = 0.25


# (x,y) - top-left corner, (w, h) - width & height of bounded box

def visualize(image: np.ndarray, boxes, scores) -> np.ndarray:
    num_boxes = len(boxes)

    if num_boxes <= 0:
        return image

    image_h, image_w, _ = image.shape

    for i in range(0, num_boxes):
        x1, y1, x2, y2 = boxes[i]
        start_point = (int(x1 * image_w), int(y1 * image_h))
        end_point = (int(x2 * image_w), int(y2 * image_h))
        cv2.rectangle(img=image, pt1=start_point, pt2=end_point, color=_BOX_COLOR, thickness=3)
        cv2.circle(img=image, center=start_point, radius=5, color=(0, 0, 0), thickness=3)  # черная
        cv2.circle(img=image, center=end_point, radius=5, color=(255, 0, 0), thickness=3)  # красная
        probability = round(scores[i, 0], 2)

        cv2.putText(image, f"Worker ({str(probability)})",
                    (start_point[0], start_point[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS, lineType=cv2.LINE_AA)

    return image


def filter_boxes(box_xywh, scores, score_threshold: float, input_shape):
    scores_max = np.amax(scores, axis=-1)
    mask = scores_max >= score_threshold

    print(f"scores_max: {scores_max.shape}")
    print(f"box_xywh: {box_xywh.shape}")

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


def run(model: str, camera_id: int):
    vid = cv2.VideoCapture(camera_id)

    interpreter = Interpreter(model_path=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    frame_id = 0

    while True:
        success, frame = vid.read()
        if not success:
            print("Video over.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()

        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        # print("BEFORE +++++++++++++++++++++")
        # print(f"Predictions shape: {pred[0].shape}")
        # print(f"Predictions: {pred[0]}")
        # print(f"Scores shape: {pred[1].shape}")
        # print(f"Scores: {pred[1]}")
        # отбираем по скору
        boxes, pred_conf = filter_boxes(box_xywh=pred[0], scores=pred[1], score_threshold=0.25,
                                        input_shape=(int(input_size), int(input_size)))
        # print("AFTER +++++++++++++++++++++")
        # print(f"Predictions shape: {boxes.shape}")
        # print(f"Predictions: {boxes}")
        # print(f"Scores shape: {pred_conf.shape}")
        # print(f"Scores: {pred_conf}")
        # non-max suppression
        best_predictions = non_max_suppression(boxes=boxes,
                                               scores=pred_conf[0, :, 0],
                                               max_bbox_overlap=0.5,
                                               frame_size=(frame.shape[0], frame.shape[1]))
        if len(best_predictions) > 0:
            logger.info(f"Worker")

        image = visualize(frame, boxes[0, best_predictions], pred_conf[0, best_predictions])

        curr_time = time.time()
        exec_time = curr_time - prev_time

        # fps
        fps = int(1 / exec_time)
        print(f"fps: {fps}")

        result = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1


if __name__ == '__main__':
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename="logs.txt", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    logger.info("Script started")
    run(model="./weights/yolov4-tiny-416-metadata.tflite", camera_id=0)
    # run(model="yolov4-tiny-416-fp16.tflite", camera_id=0)
