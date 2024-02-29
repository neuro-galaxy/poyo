import math
import logging
from typing import List, Dict, Tuple, Optional
from functools import cached_property

import torch

from poyo.data.dataset import DatasetIndex


class RandomFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the 
    :obj:`interval_dict` parameter. :obj:`interval_dict` is a dictionary where the keys 
    are the session ids and the values are lists of tuples representing the 
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:
    
    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        interval_dict (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        interval_dict: Dict[str, List[Tuple[int, int]]],
        window_length: float,
        generator: Optional[torch.Generator],
        drop_short: bool = True,
    ):
        self.interval_dict = interval_dict
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.
        
        for session_name, sampling_intervals in self.interval_dict.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError (
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                            )

                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Will train on {num_samples * self.window_length} seconds."
                )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        return self._estimated_len

    def __iter__(self):
        indices = []
        for session_name, sampling_intervals in self.interval_dict.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        continue
                    else:
                        raise ValueError (
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                            )

                # sample a random offset
                left_offset = (
                    torch.rand(1, generator=self.generator).item() * self.window_length
                )

                indices_ = [
                        DatasetIndex(
                            session_name, t.item(), (t + self.window_length).item()
                        )
                        for t in torch.arange(start + left_offset, end, self.window_length, dtype=torch.float64)
                        if t + self.window_length <= end
                    ]
                
                if len(indices_) > 0:
                    indices.extend(indices_)
                    right_offset = end - indices[-1].end
                else:
                    right_offset = end - start - left_offset

                # if there is one sample worth of data, add it
                # this ensures that the number of samples is always consistent
                if right_offset + left_offset >= self.window_length:
                    if right_offset > left_offset:
                        indices.append(
                            DatasetIndex(session_name, end - self.window_length, end)
                        )
                    else:
                        indices.append(
                            DatasetIndex(
                                session_name, start, start + self.window_length
                            )
                        )

        # shuffle
        for idx in torch.randperm(len(indices), generator=self.generator):
            yield indices[idx]


class SequentialFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows sequentially, always in the same order. The
    sampling intervals are defined in the :obj:`interval_dict` parameter. 
    :obj:`interval_dict` is a dictionary where the keys are the session ids and the 
    values are lists of tuples representing the start and end of the intervals 
    from which to sample.

    If the length of a sequence is not evenly divisible by the step, the last
    window will be added with an overlap with the previous window. This is to ensure
    that the entire sequence is covered.

    Args:
        interval_dict (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        step (Optional[float], optional): Step size between windows. If None, it
            defaults to `window_length`. Defaults to None.
    """

    def __init__(
        self,
        *,
        interval_dict: Dict[str, List[Tuple[int, int]]],
        window_length: float,
        step: Optional[float] = None,
        drop_short=False,
    ):
        self.interval_dict = interval_dict
        self.window_length = window_length
        self.step = step or window_length

        assert self.step > 0, "Step must be greater than 0."
        assert self.step <= self.window_length, "Step must be less than window length."

    # we cache the indices since they are deterministic
    @cached_property
    def _indices(self) -> List[DatasetIndex]:
        indices = []
        for session_name, sampling_intervals in self.interval_dict.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                assert interval_length >= self.window_length, (
                    f"Interval {(start, end)} is too short to sample from. "
                    f"Minimum length is {self.window_length}."
                )

                indices_ = [
                        DatasetIndex(
                            session_name, t.item(), (t + self.window_length).item()
                        )
                        for t in torch.arange(start, end, self.step, dtype=torch.float64)
                        if t + self.window_length <= end
                ]

                indices.extend(indices_)

                # we need to make sure that the entire interval is covered
                if indices_[-1].end < end:
                    indices.append(
                        DatasetIndex(session_name, end - self.window_length, end)
                    )

        return indices

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        yield from self._indices
