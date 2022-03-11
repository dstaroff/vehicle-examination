import datetime
from typing import (
    Tuple,
    Type,
    Union,
    Optional,
    List,
)

import numpy as np
import qimage2ndarray
from PySide2.QtCore import (
    QRunnable,
    QThreadPool,
    Signal,
)
from PySide2.QtGui import (
    QPixmap, QPalette, QColor, Qt,
)
from PySide2.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from cv2 import cv2

from src.capture import (
    Camera,
    Video,
)
from src.capture.base import (
    CaptureSource,
    CaptureSourceImpl,
)
from src.detector import SubjectDetector
from src.detector.examination import (
    ExaminationDetector,
    ExaminationState,
)
from src.detector.subject.model import (
    SubjectImpl,
    GaplessSubject,
)
from src.detector.subject.model import (
    Vehicle,
    Person,
)
from src.utils import const

radius = 5
frame_count = 0
stop_needed = False


# noinspection PyAttributeOutsideInit
class App(QApplication):
    _update_pixmap_signal: Signal = Signal()
    _update_examination_state_signal: Signal = Signal()

    def __init__(self):
        super(App, self).__init__()
        self.setApplicationName(const.APP_NAME)
        self.setApplicationDisplayName(const.APP_DISPLAY_NAME)
        self.aboutToQuit.connect(lambda: self.force_stop())
        self.setQuitOnLastWindowClosed(True)
        self._window = QMainWindow()

        self._debug_mode = False
        self._subject_detector = SubjectDetector()
        self._examination_detector = ExaminationDetector()

        self._examination_state: ExaminationState = ExaminationState.NoVehicle

        self._capture: Optional[CaptureSource] = None
        self._pixmap: Optional[QPixmap] = None
        self._last_frame_time: Optional[datetime.datetime] = None

        self._gapless_vehicle = GaplessSubject()
        self._gapless_examiner = GaplessSubject()

        self._setup_ui()

    @staticmethod
    def force_stop():
        global stop_needed

        stop_needed = True

    def _set_debug_mode(self, debug_mode: bool):
        self._debug_mode = debug_mode
        if self._debug_mode:
            self.setApplicationDisplayName(f'{const.APP_DISPLAY_NAME} [DEBUG MODE]')
        else:
            self.setApplicationDisplayName(const.APP_DISPLAY_NAME)

    def _setup_ui(self):
        self._setup_menu()

        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)

        video_control_container = QWidget()
        video_control_layout = QHBoxLayout(video_control_container)
        video_control_layout.addLayout(self._setup_video_ui())

        info_container = QWidget()
        info_layout = QHBoxLayout(info_container)
        self._examination_state_box = self._setup_examination_box()
        info_layout.addWidget(self._examination_state_box)

        main_layout.addWidget(video_control_container)
        main_layout.addWidget(info_container)

        self._window.setCentralWidget(main_container)

    def _setup_menu(self):
        menu = self._window.menuBar()
        capture_menu = menu.addMenu('Capture')
        utils_menu = menu.addMenu('Utils')

        camera_capture = QAction('From Camera', self._window)
        camera_capture.setShortcut('Ctrl+C')
        # noinspection PyUnresolvedReferences
        camera_capture.triggered.connect(lambda: self.set_capture(Camera(0)))

        video_capture = QAction('From Video', self._window)
        video_capture.setShortcut('Ctrl+V')
        # noinspection PyUnresolvedReferences
        video_capture.triggered.connect(self._set_video_capture)

        debug = QAction('Debug mode', self._window)
        debug.setShortcut('Ctrl+D')
        # noinspection PyUnresolvedReferences
        debug.triggered.connect(lambda: self._set_debug_mode(not self._debug_mode))

        capture_menu.addActions([camera_capture, video_capture])
        utils_menu.addAction(debug)

    def _setup_video_ui(self) -> QLayout:
        layout = QHBoxLayout()
        self._video = QLabel('No Camera Feed')
        layout.addWidget(self._video)

        return layout

    def _setup_examination_box(self) -> QCheckBox:
        examination_state_box = QCheckBox(f'Examination: {self._examination_state.value}')
        examination_state_box.setEnabled(False)
        pal = examination_state_box.palette()
        pal.setColor(QPalette.Inactive, QPalette.WindowText, QColor(Qt.black))
        examination_state_box.setPalette(pal)
        examination_state_box.setStyleSheet(
            '''
            QCheckBox:disabled {
                border: none;
                color: black;
                background: none;
            }
            QCheckBox::text:disabled {
                border: none;
                color: black;
            }
            QCheckBox::indicator:checked:disabled { background-color: #00FF00; color: #00FF00; }
            QCheckBox::indicator:unchecked:disabled { background-color: #FF0000; color: #FF0000;}
            ''',
        )
        examination_state_box.update()

        return examination_state_box

    def set_capture(self, capture_source: Union[Type[CaptureSource], CaptureSourceImpl]):
        if self._capture is not None:
            self._capture.release()

        self._capture = capture_source

    def _set_video_capture(self):
        file_name = QFileDialog.getOpenFileName(
            self._window,
            'Choose a video to capture from...',
            const.PROJECT_PATH,
            'Videos (*.mp4)'
        )[0]
        if file_name:
            self.set_capture(Video(file_name))

    def _update_examination_state(self):
        self._examination_state_box.setText(f'Examination: {self._examination_state.value}')
        check = True if self._examination_state == ExaminationState.Examined else False
        self._examination_state_box.setChecked(check)

    # noinspection PyBroadException
    def process(self):
        if self._capture is None:
            return

        global frame_count

        try:
            self._capture.skip_frames(self._calculate_skip_frames())

            frame = self._capture.read()
            subjects = self._filter_biggest_subjects(self._subject_detector.detect(frame))

            for subject in subjects:
                if self._debug_mode:
                    frame = self._draw_mask(subject.mask(), frame, subject.color())

                    distance = int(ExaminationDetector.get_max_distance(self._capture.frame_size()))
                    cv2.circle(
                        frame,
                        (int(subject.centroid().x), int(subject.centroid().y)),
                        distance,
                        const.COLOR_WHITE,
                    )
                    cv2.rectangle(
                        frame,
                        (int(subject.bounding_box().p1.x), int(subject.bounding_box().p1.y)),
                        (int(subject.bounding_box().p2.x), int(subject.bounding_box().p2.y)),
                        const.COLOR_RED,
                    )
                    cv2.rectangle(
                        frame,
                        (
                            int(subject.bounding_box().p1.x) - distance,
                            int(subject.bounding_box().p1.y) - distance,
                        ),
                        (
                            int(subject.bounding_box().p2.x) + distance,
                            int(subject.bounding_box().p2.y) + distance,
                        ),
                        const.COLOR_GREEN,
                    )

                text = f'{subject.caption()}'

                text_coords = (int(subject.centroid().x) - 2 * radius,
                               int(subject.centroid().y) - 2 * radius)

                cv2.putText(frame, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, const.COLOR_BLACK, 5)
                cv2.putText(frame, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, const.COLOR_WHITE, 1)

            try:
                self._examination_state = self._examination_detector.detect(subjects, self._capture.frame_size())
            except Exception:
                pass

            self._update_examination_state_signal.emit()

            image = qimage2ndarray.array2qimage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._pixmap = QPixmap.fromImage(image)
            # noinspection PyUnresolvedReferences
            self._update_pixmap_signal.emit()
        except Exception:
            pass

        frame_count += 1

    def _filter_biggest_subjects(self, subjects: List[SubjectImpl]) -> List[SubjectImpl]:
        vehicle: Optional[Vehicle] = ExaminationDetector.get_biggest_subject_by_type(subjects, Vehicle)
        examiner: Optional[Person] = ExaminationDetector.get_biggest_subject_by_type(subjects, Person)

        self._gapless_vehicle.add_subject(vehicle)
        self._gapless_examiner.add_subject(examiner)

        vehicle = self._gapless_vehicle.get_subject()
        examiner = self._gapless_examiner.get_subject()

        res = []
        if vehicle:
            res.append(vehicle)
        if examiner:
            res.append(examiner)

        return res

    @staticmethod
    def _draw_mask(mask: np.ndarray, frame: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        for c in range(frame.shape[2]):
            frame[:, :, c] = np.where(
                mask > 0,
                frame[:, :, c] * (1 - const.FIELD_OPACITY) + color[c] * const.FIELD_OPACITY,
                frame[:, :, c]
            )

        return frame

    def _update_pixmap(self):
        if self._pixmap is not None:
            self._video.setPixmap(self._pixmap)

    def _calculate_skip_frames(self) -> int:
        current_time = datetime.datetime.now()

        frames = 0

        if self._last_frame_time is not None:
            time_delta = current_time - self._last_frame_time
            frames = round((self._capture.fps() * time_delta.microseconds) / 1_000_000)

        self._last_frame_time = current_time

        return frames

    def run(self):
        # noinspection PyUnresolvedReferences
        self._update_examination_state_signal.connect(self._update_examination_state)
        # noinspection PyUnresolvedReferences
        self._update_pixmap_signal.connect(self._update_pixmap)

        processing = VideoProcessing(self)
        processing.setAutoDelete(True)

        pool = QThreadPool.globalInstance()

        self._window.show()
        pool.start(processing)
        return self.exec_()


class VideoProcessing(QRunnable):
    def __init__(self, app: App):
        super().__init__()
        self._app = app

    def run(self):
        while not stop_needed:
            self._app.process()
