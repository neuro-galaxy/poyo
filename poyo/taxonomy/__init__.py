from .core import StringIntEnum, Dictable

from .subject import (
    Species,
    Sex,
)

from .task import (
    Task,
)

from .macaque import Macaque

from .recording_tech import (
    RecordingTech,
)

from .descriptors import (
    DandisetDescription,
    SubjectDescription,
    SortsetDescription,
    SessionDescription,
    to_serializable,
)

class OutputType(StringIntEnum):
    CONTINUOUS = 0
    BINARY = 1
    MULTILABEL = 2
    MULTINOMIAL = 3
