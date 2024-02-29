from dataclasses import dataclass

from .core import StringIntEnum, Dictable


class RecordingTech(StringIntEnum):
    UTAH_ARRAY_SPIKES = 0
    UTAH_ARRAY_THRESHOLD_CROSSINGS = 1
