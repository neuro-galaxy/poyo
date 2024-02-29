from .core import StringIntEnum


class Task(StringIntEnum):
    REACHING = 0


class REACHING(StringIntEnum, parent=Task.REACHING):
    """A classic BCI task involving reaching to a 2d target."""
    RANDOM = 0
    HOLD = 1
    REACH = 2
    RETURN = 3
    INVALID = 4
    OUTLIER = 5
