from collections import deque
from typing import Deque, Optional

from .subject import SubjectImpl


class GaplessSubject:
    MAX_COUNT: int = 10

    def __init__(self):
        self._deque: Deque[Optional[SubjectImpl]] = deque([], maxlen=self.MAX_COUNT)

    def get_subject(self) -> Optional[SubjectImpl]:
        res = None
        for subject in self._deque:
            if subject:
                res = subject

        return res

    def add_subject(self, subject: SubjectImpl):
        self._deque.append(subject)
