FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)
BOX_COLOR = (0, 255, 0)

INPUT_SIZE = 416  # input size of tensors for model
IOU_THRESHOLD = 0.45  # for non-max-suppresion
SCORE_THRESHOLD = 0.7  # score threshold to show detection

MODEL_PATH = "./weights/yolov4-tiny-416-metadata.tflite"
LOG_FILE_NAME = "logs.txt"
SOURCE = 0  # path to video or camera_id
