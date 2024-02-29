import collections
import dataclasses
import datetime
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from pydantic.dataclasses import dataclass

from .core import Dictable, StringIntEnum
from poyo.taxonomy import *


@dataclass
class SessionDescription(Dictable):
    id: str 
    recording_date: datetime.datetime
    task: Task
    # Fields below are automatically filled by the SessionContextManager
    # Do not set them manually.
    splits: Dict[str, List[Tuple[float, float]]] = None # should be filled by register_split() only
    dandiset_id: Optional[str] = None
    subject_id: Optional[str] = None
    sortset_id: Optional[str] = None


@dataclass
class SortsetDescription(Dictable):
    id: str
    units: List[str]
    subject: str
    areas: Union[List[StringIntEnum], List[Macaque]]
    recording_tech: List[RecordingTech]
    # Fields below are automatically filled by the SessionContextManager
    # Do not set them manually.
    sessions: List[SessionDescription] = None # should be filled by register_session() only


@dataclass
class SubjectDescription(Dictable):
    id: str
    species: Species
    age: float = 0.0  # in days
    sex: Sex = Sex.UNKNOWN
    genotype: str = "unknown"  # no idea how many there will be for now.


@dataclass
class DandisetDescription(Dictable):
    id: str
    origin_version: str
    derived_version: str
    metadata_version: str
    source: str
    description: str
    splits: List[str]
    subjects: List[SubjectDescription]
    sortsets: List[SortsetDescription]


def to_serializable(dct):
    """Recursively map data structure elements to string when they are of type
    StringIntEnum"""
    if isinstance(dct, list) or isinstance(dct, tuple):
        return [to_serializable(x) for x in dct]
    elif isinstance(dct, dict) or isinstance(dct, collections.defaultdict):
        return {
            to_serializable(x): to_serializable(y)
            for x, y in dict(dct).items()
        }
    elif isinstance(dct, Dictable):
        return {
            x.name: to_serializable(getattr(dct, x.name))
            for x in dataclasses.fields(dct)
        }
    elif isinstance(dct, StringIntEnum):
        return str(dct)
    elif isinstance(dct, np.ndarray):
        if np.isscalar(dct):
            return dct.item()
        else:
            raise NotImplementedError("Cannot serialize numpy arrays.")
    elif (
        isinstance(dct, str)
        or isinstance(dct, int)
        or isinstance(dct, float)
        or isinstance(dct, bool)
        or isinstance(dct, type(None))
        or isinstance(dct, datetime.datetime)
    ):
        return dct
    else:
        raise NotImplementedError(f"Cannot serialize {type(dct)}")
