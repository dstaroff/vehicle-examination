import math
from typing import List, Optional, Type

from src.capture.model import FrameSize
from src.detector.subject.model import SubjectImpl, Vehicle, Person
from src.gui.model.geometry import Point, Rectangle
from .state import ExaminationState


class ExaminationDetector:
    def __init__(self):
        self._last_frame_vehicle: Optional[Vehicle] = None
        self._last_frame_vehicle_state: ExaminationState = ExaminationState.NoVehicle

    def return_state_with_save(self, state: ExaminationState) -> ExaminationState:
        self._last_frame_vehicle_state = state
        return state

    def detect(self, subjects: List[SubjectImpl], frame_size: FrameSize) -> ExaminationState:
        vehicle: Optional[Vehicle] = self.get_biggest_subject_by_type(subjects, Vehicle)
        examiner: Optional[Person] = self.get_biggest_subject_by_type(subjects, Person)

        if not vehicle:
            return self.return_state_with_save(ExaminationState.NoVehicle)
        elif self._last_frame_vehicle_state == ExaminationState.NoVehicle:
            self._last_frame_vehicle_state = ExaminationState.NotExamined

        state = self._decide_vehicle_state(vehicle, frame_size)
        self._last_frame_vehicle = vehicle

        if not examiner:
            return self.return_state_with_save(state)

        state = self._decide_examination(vehicle, examiner, frame_size)

        return self.return_state_with_save(state)

    @staticmethod
    def get_biggest_subject_by_type(subjects: List[SubjectImpl], subject_type: Type[SubjectImpl]) -> Optional[SubjectImpl]:
        res: Optional[SubjectImpl] = None

        biggest_area = 0
        for subject in subjects:
            if type(subject) != subject_type:
                continue

            subject_area = subject.mask_area()
            if subject_area > biggest_area:
                res = subject
                biggest_area = subject_area

        return res

    def _decide_vehicle_state(self, current_frame_vehicle: Vehicle, frame_size: FrameSize) -> ExaminationState:
        if self._last_frame_vehicle:
            distance = l2(self._last_frame_vehicle.centroid(), current_frame_vehicle.centroid())
            if distance <= self.get_max_distance(frame_size):
                return self._last_frame_vehicle_state

        return ExaminationState.NotExamined

    @staticmethod
    def _decide_examination(vehicle: Vehicle, examiner: Person, frame_size: FrameSize) -> ExaminationState:
        distance = rect_l2(vehicle.bounding_box(), examiner.bounding_box())
        if distance <= ExaminationDetector.get_max_distance(frame_size):
            return ExaminationState.Examined

        return ExaminationState.NotExamined

    @staticmethod
    def get_max_distance(frame_size: FrameSize) -> float:
        return min(frame_size.width, frame_size.height) / 4


def l2(centroid1: Point, centroid2: Point) -> float:
    return math.sqrt(math.pow(centroid1.x - centroid2.x, 2.0) + math.pow(centroid1.y - centroid2.y, 2.0))


def rect_l2(rect1: Rectangle, rect2: Rectangle) -> float:
    left = rect2.p2.x < rect1.p1.x
    right = rect1.p2.x < rect2.p1.x
    bottom = rect2.p2.y < rect1.p1.y
    top = rect1.p2.y < rect2.p1.y

    if top and left:
        return l2(
            Point(rect1.p1.x, rect1.p2.y),
            Point(rect2.p2.x, rect2.p1.y),
        )
    elif left and bottom:
        return l2(
            Point(rect1.p1.x, rect1.p1.y),
            Point(rect2.p2.x, rect2.p2.y),
        )
    elif bottom and right:
        return l2(
            Point(rect1.p2.x, rect1.p1.y),
            Point(rect2.p1.x, rect2.p2.y),
        )
    elif right and top:
        return l2(
            Point(rect1.p2.x, rect1.p2.y),
            Point(rect2.p1.x, rect2.p1.y),
        )
    elif left:
        return rect1.p1.x - rect2.p2.x
    elif right:
        return rect2.p1.x - rect1.p2.x
    elif bottom:
        return rect1.p1.y - rect2.p2.y
    elif top:
        return rect2.p1.y - rect1.p2.y
    else:
        return 0.0
