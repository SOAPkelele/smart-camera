import cv2
import numpy as np

from app.config import BOX_COLOR, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS


def visualize(image: np.ndarray, boxes, scores) -> np.ndarray:
    num_boxes = len(boxes)

    if num_boxes <= 0:
        return image

    image_h, image_w, _ = image.shape

    for i in range(0, num_boxes):
        x1, y1, x2, y2 = boxes[i]
        start_point = (int(x1 * image_w), int(y1 * image_h))
        end_point = (int(x2 * image_w), int(y2 * image_h))
        cv2.rectangle(img=image, pt1=start_point, pt2=end_point, color=BOX_COLOR, thickness=3)
        cv2.circle(img=image, center=start_point, radius=5, color=(0, 0, 0), thickness=3)  # черная
        cv2.circle(img=image, center=end_point, radius=5, color=(255, 0, 0), thickness=3)  # красная
        probability = round(scores[i, 0], 2)

        cv2.putText(image, f"Worker ({str(probability)})",
                    (start_point[0], start_point[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, lineType=cv2.LINE_AA)

    return image
