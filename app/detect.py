import time

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

from app.config import INPUT_SIZE, SCORE_THRESHOLD, IOU_THRESHOLD, MODEL_PATH, LOG_FILE_NAME, SOURCE
from app.utils import filter_boxes, non_max_suppression, visualize, setup_logger


def run(model: str, camera_id: int):
    vid = cv2.VideoCapture(camera_id)

    interpreter = Interpreter(model_path=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        success, frame = vid.read()
        if not success:
            logger.info("Video over.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()

        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        boxes, pred_conf = filter_boxes(box_xywh=pred[0], scores=pred[1], score_threshold=SCORE_THRESHOLD,
                                        input_shape=(int(INPUT_SIZE), int(INPUT_SIZE)))

        best_predictions = non_max_suppression(boxes=boxes,
                                               scores=pred_conf[0, :, 0],
                                               max_bbox_overlap=IOU_THRESHOLD,
                                               frame_size=(frame.shape[0], frame.shape[1]))

        num_workers = len(boxes[0, best_predictions])
        if num_workers > 0:
            logger.info(f"Worker" if num_workers == 1 else f"{num_workers} workers")

        image = visualize(frame, boxes[0, best_predictions], pred_conf[0, best_predictions])

        fps = int(1 / time.time() - prev_time)

        result = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    logger = setup_logger(LOG_FILE_NAME)
    run(model=MODEL_PATH, camera_id=SOURCE)
