from typing import List

import imutils
import numpy as np
from cv2 import cv2

from src.capture.model import FrameSize
from src.gui.model.geometry import Point
from src.gui.model.geometry import Rectangle
from src.utils import const
from .mask_rcnn.config import Config
from .mask_rcnn.mrcnn import MaskRCNN
from .mask_rcnn.utils import resize
from .model import (
    subject_from_class_id,
    SubjectImpl,
)


class SubjectDetector:
    def __init__(self):
        self._model = MaskRCNN(Config(len(const.CLASS_NAMES_ALL)))
        self._model.load_weights(const.MODEL_WEIGHTS_PATH, by_name=True)

    @staticmethod
    def _scale_rectangle_coords(rectangle: Rectangle, x_scale_factor: float, y_scale_factor: float) -> Rectangle:
        return Rectangle(
            p1=SubjectDetector._scale_point_coords(rectangle.p1, x_scale_factor, y_scale_factor),
            p2=SubjectDetector._scale_point_coords(rectangle.p2, x_scale_factor, y_scale_factor),
        )

    @staticmethod
    def _roi_to_rectangle(roi: List[int]) -> Rectangle:
        return Rectangle(
            p1=Point(
                x=roi[1],
                y=roi[0],
            ),
            p2=Point(
                x=roi[3],
                y=roi[2]
            ),
        )

    @staticmethod
    def _scale_point_coords(point: Point, x_scale_factor: float, y_scale_factor: float) -> Point:
        return Point(point.x * x_scale_factor, point.y * y_scale_factor)

    @staticmethod
    def _roi_to_centroid(roi: List[int]) -> Point:
        return Point(
            x=roi[1] + (roi[3] - roi[1]) // 2,
            y=roi[0] + (roi[2] - roi[0]) // 2,
        )

    @staticmethod
    def _scale_mask(mask: np.ndarray, image_shape: FrameSize) -> np.ndarray:
        """
        Converts a mask returned by Mask-RCNN to frame original shape.
        Returns a monochrome image mask with the same size as the original image,
        where 0 means NO SUBJECT and 255 means SUBJECT
        """
        int_mask = mask.astype(dtype='int8')
        resized_mask = resize(int_mask, (image_shape.height, image_shape.width))
        image_mask = np.where(resized_mask > 0, True, False).astype(np.bool)

        return image_mask

    def detect(self, frame: np.ndarray) -> List[SubjectImpl]:
        scale_factor = frame.shape[1] / const.MAX_MASKRCNN_WIDTH

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = imutils.resize(img, width=const.MAX_MASKRCNN_WIDTH, inter=cv2.INTER_NEAREST)

        detections = self._model.detect([img])[0]

        num_detections = len(detections['rois'])

        res = []
        for i in range(num_detections):
            subject = subject_from_class_id(int(detections['class_ids'][i]))
            if subject is not None:
                subject.set_score(detections['scores'][i])
                subject.set_bounding_box(
                    self._scale_rectangle_coords(
                        self._roi_to_rectangle(
                            list(detections['rois'][i]),
                        ),
                        scale_factor, scale_factor,
                    )
                )
                subject.set_centroid(
                    self._scale_point_coords(
                        self._roi_to_centroid(
                            list(detections['rois'][i]),
                        ),
                        scale_factor, scale_factor,
                    )
                )
                subject.set_mask(
                    self._scale_mask(
                        detections['masks'][:, :, i],
                        FrameSize(width=frame.shape[1], height=frame.shape[0]),
                    )
                )
                res.append(subject)

        return res
