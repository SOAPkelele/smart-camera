_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 0)
_BOX_COLOR = (0, 255, 0)

input_size = 416  # input size of tensors for model
iou = 0.45  # for non-max-suppresion
score = 0.7  # score threshold to show detection

MODEL_PATH = "./weights/yolov4-tiny-416-metadata.tflite"
LOG_FILE_NAME = "logs.txt"
SOURCE = 0  # path to video or camera_id
