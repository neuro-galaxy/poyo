import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import datetime

import msgpack
import h5py
import torch

from poyo.data import Data, Interval


@dataclass
class DatasetIndex:
    """Accessing the dataset is done by specifying a session id and a time interval."""
    session_id: str
    start: float
    end: float


class Dataset(torch.utils.data.Dataset):
    r"""This class abstracts a collection of lazily-loaded Data objects. Each of these
    Data objects corresponds to a session and lives on the disk until it is requested.
    The `include` argument guides which sessions are included in this Dataset.
    To request a piece of a included session's data, you can use the `get` method,
    or index the Dataset with a `DatasetIndex` object (see `__getitem__`).

    This definition is a deviation from the standard PyTorch Dataset definition, which
    generally presents the dataset directly as samples. In this case, the Dataset
    by itself does not provide you with samples, but rather the means to flexibly work
    and accesss complete sessions.
    Within this framework, it is the job of the sampler to provide the
    DatasetIndex indices to slice the dataset into samples (see `poyo.data.sampler`).

    Args:
        root: The root directory of the dataset.
        split: The split of the dataset. This is used to determine the sampling intervals
            for each session.
        include: A list of dictionaries specifying the datasets to include. Each dictionary
            should have the following keys:
            - dandiset: The dandiset to include.
            - selection: A dictionary specifying the selection criteria for the dataset.
        transform: A transform to apply to the data. This transform should be a callable
            that takes a Data object and returns a Data object.
        keep_files_open: If True, the files are kept open and the data is loaded into
            memory. This is useful for efficiency. If False, the files are opened and
            closed every time a piece of data is requested. This is useful for memory
            efficiency.
    """

    _check_for_data_leakage_flag: bool = True
    _open_files: Optional[Dict[str, h5py.File]] = None
    _data_objects: Optional[Dict[str, Data]] = None

    def __init__(
        self,
        root: str,
        split: str,
        include: List[Dict[str, Any]],
        transform=None,
        keep_files_open: bool = True,
    ):
        super().__init__()
        self.root = root
        self.split = split

        if include is None:
            raise ValueError("Please specify the datasets to include")

        self.include = include
        self.transform = transform

        self.session_info_dict, self.session_ids, self.unit_ids = self._look_for_files()

        if keep_files_open:
            self._open_files = {
                session_id: h5py.File(session_info["filename"], "r")
                for session_id, session_info in self.session_info_dict.items()
            }

            self._data_objects = {
                session_id: Data.from_hdf5(f)
                for session_id, f in self._open_files.items()
            }

    def _close_open_files(self):
        """Closes the open files and deletes open data objects.
        This is useful when you are done with the dataset.
        """
        if self._open_files is not None:
            for f in self._open_files.values():
                f.close()
            self._open_files = None

        self._data_objects = None  # initialized Data objects should be gc'd

    def __del__(self):
        self._close_open_files()

    def _look_for_files(self) -> Tuple[Dict[str, Dict], List[str], List[str]]:
        session_info_dict = {}
        session_ids = []
        unit_ids = []

        for i, selection_list in enumerate(self.include):
            selection = selection_list["selection"]
            config = selection_list.get("config", {})

            # parse selection
            if len(selection) == 0:
                raise ValueError(
                    f"Selection {i} is empty. Please at least specify a dandiset."
                )

            for subselection in selection:
                if subselection.get("dandiset", "") == "":
                    raise ValueError(f"Please specify a dandiset to include.")

                description_file = os.path.join(
                    self.root, subselection["dandiset"], "description.mpk"
                )

                try:
                    with open(description_file, "rb") as f:
                        description = msgpack.load(
                            f, object_hook=decode_datetime
                        )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Could not find description file {description_file}. This error "
                        "might be due to running an old pipeline that generates a "
                        "description.yaml file. Try running the appropriate snakemake "
                        "pipeline to generate the msgpack (mpk) file instead."
                    )

                # Get a list of all the potentially chunks in this dataset.
                sortsets = description["sortsets"]
                all_sortset_ids = [x["id"] for x in sortsets]
                all_sortset_subjects = set([x["subject"] for x in sortsets])

                # Perform selection. Right now, we are limiting ourselves to sortset,
                # subject and session, but we could make selection more flexible in the
                # future.
                sel_sortset = subselection.get("sortset", None)
                sel_sortsets = subselection.get("sortsets", None)
                sel_sortset_lte = subselection.get("sortset_lte", None)
                sel_subject = subselection.get("subject", None)
                sel_subjects = subselection.get("subjects", None)
                # exclude_sortsets allows you to exclude some sortsets from the selection.
                # example use: you want to train on the complete dandiset, but leave out
                # a few sortsets for evaluating transfer performance.
                sel_exclude_sortsets = subselection.get("exclude_sortsets", None)

                sel_session = subselection.get("session", None)
                sel_output = subselection.get("output", None)

                filtered = False
                if sel_sortset is not None:
                    assert (
                        sel_sortset in all_sortset_ids
                    ), f"Sortset {sel_sortset} not found in dandiset {subselection['dandiset']}"
                    sortsets = [
                        sortset for sortset in sortsets if sortset["id"] == sel_sortset
                    ]
                    filtered = True

                if sel_sortsets is not None:
                    assert (
                        not filtered
                    ), "Cannot specify sortset AND sortsets in selection"

                    # Check that all sortsets are in the dandiset.
                    for sortset in sel_sortsets:
                        assert (
                            sortset in all_sortset_ids
                        ), f"Sortset {sortset} not found in dandiset {subselection['dandiset']}"

                    sortsets = [
                        sortset for sortset in sortsets if sortset["id"] in sel_sortsets
                    ]
                    filtered = True

                if sel_sortset_lte is not None:
                    assert (
                        not filtered
                    ), "Cannot specify sortset_lte AND sortset(s) in selection"

                    sortsets = [
                        sortset
                        for sortset in sortsets
                        if sortset["id"] <= sel_sortset_lte
                    ]
                    filtered = True

                if sel_subject is not None:
                    assert (
                        not filtered
                    ), "Cannot specify subject AND sortset(s)/sortset_lte in selection"

                    assert (
                        sel_subject in all_sortset_subjects
                    ), f"Could not find subject {sel_subject} in dandiset {subselection['dandiset']}"

                    sortsets = [
                        sortset
                        for sortset in sortsets
                        if sortset["subject"] == sel_subject
                    ]
                    filtered = True

                if sel_subjects is not None:
                    assert (
                        not filtered
                    ), "Cannot specify subjects AND subject/sortset(s)/sortset_lte in selection"

                    # Make sure all subjects asked for are in the dandiset
                    sel_subjects = set(sel_subjects)
                    assert sel_subjects.issubset(all_sortset_subjects), (
                        f"Could not find subject(s) {sel_subjects - all_sortset_subjects} "
                        f" in dandiset {subselection['dandiset']}"
                    )

                    sortsets = [
                        sortset
                        for sortset in sortsets
                        if sortset["subject"] in sel_subjects
                    ]
                    filtered = True

                # Exclude sortsets if asked.
                if sel_exclude_sortsets is not None:
                    sortsets = [
                        sortset
                        for sortset in sortsets
                        if sortset["id"] not in sel_exclude_sortsets
                    ]

                # Note that this logic may result in adding too many slots but that's fine.
                unit_ids += [x for sortset in sortsets for x in sortset["units"]]
                # unit_ids are already fully qualified with prepended dandiset id.

                # Now we get the session-level information.
                sessions = sum([sortset["sessions"] for sortset in sortsets], [])
                if sel_session is not None:
                    sessions = [
                        session for session in sessions if session["id"] == sel_session
                    ]

                assert (
                    len(sessions) > 0
                ), f"No sessions found for {i}'th selection included"

                # Similarly, select for certain outputs
                if sel_output is not None:
                    sessions = [
                        session
                        for session in sessions
                        if sel_output in session["fields"].keys()
                    ]

                # Now we get the session-level information
                for session in sessions:
                    # iomap = {k: session[k] for k in ["fields", "task"]}

                    # # Check that the chunk has the requisite inputs.
                    # check = check_include(selection_list, iomap["fields"])
                    # if not check:
                    #     continue

                    session_id = subselection["dandiset"] + "/" + session["id"]

                    if session_id in session_info_dict:
                        raise ValueError(
                            f"Session {session_id} is already included in the dataset."
                            "Please verify that it is only selected once."
                        )
                    
                    session_ids.append(session_id)

                    session_info_dict[session_id] = dict(
                        filename=(Path(self.root) / (session_id + ".h5")),
                        sampling_intervals=Interval.from_list(session["splits"][self.split]),
                        config=config,
                    )

        unit_ids = list(set(unit_ids))
        return session_info_dict, session_ids, unit_ids

    def get(self, session_id: str, start: float, end: float):
        r"""This is the main method to extract a slice from a session. It returns a
        Data object that contains all data for session :obj:`session_id` between
        times :obj:`start` and :obj:`end`.

        Args:
            session_id: The session id of the slice. Note this is the fully qualified
                session-id: <dandiset>/<session_id>
            start: The start time of the slice.
            end: The end time of the slice.
        """
        session_info = self.session_info_dict[session_id]
        if self._data_objects is None:
            with h5py.File(session_info['filename'], "r") as f:
                data = Data.from_hdf5(f)
                sample = data.slice(start, end)
        else:
            data = self._data_objects[session_id]
            sample = data.slice(start, end)

        if self._check_for_data_leakage_flag:
            sample._check_for_data_leakage(self.split)

        sample.session = session_id
        sample.config = session_info["config"]
        return sample

    def get_sampling_intervals(self):
        r"""Returns a dictionary of interval-list for each session.
        Each interval-list is a list of tuples (start, end) for each interval. This
        represents the intervals that can be sampled from each session.

        Note that these intervals will change depending on the split.
        """
        intervals = {}
        for session_id, session_info in self.session_info_dict.items():
            intervals[session_id] = list(
                zip(
                    session_info["sampling_intervals"].start,
                    session_info["sampling_intervals"].end,
                )
            )
        return intervals

    def disable_data_leakage_check(self):
        r"""Disables the data leakage check.

        .. warning::
            Only do this you are absolutely sure that there is no leakage between the
            current split and other splits (eg. the test split).
        """
        self._check_for_data_leakage_flag = False
        logging.warn(
            f"Data leakage check is disabled. Please be absolutely sure that there is "
            f"no leakage between {self.split} and other splits."
        )

    def __getitem__(self, index: DatasetIndex):
        sample = self.get(index.session_id, index.start, index.end)

        # apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        raise NotImplementedError("Length of dataset is not defined")

    def __iter__(self):
        raise NotImplementedError("Iteration over dataset is not defined")


def decode_datetime(obj):
    """msgpack doesn't support datetime, so we need to encode it as a string."""
    if "__datetime__" in obj:
        return datetime.datetime.fromisoformat(obj["as_str"])
    return obj
