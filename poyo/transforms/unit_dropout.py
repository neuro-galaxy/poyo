import numpy as np


from poyo.utils import logging

log = logging(header="UNIT DROPOUT", header_color="cyan")


class UnitCustomDistribution:
    r"""Triangular distribution with a peak at mode_units, going from min_units to 
    max_units.
    """
    def __init__(
        self,
        min_units=20,
        mode_units=100,
        max_units=300,
        peak=4,
        M=10,
        max_attempts=100,
        seed=None,
    ):
        super().__init__()
        self.min_units = min_units
        self.mode_units = mode_units
        self.max_units = max_units
        self.peak = peak
        self.M = M
        self.max_attempts = max_attempts

        self.rng = np.random.default_rng(seed=seed)

    def unnormalized_density_function(self, x):
        if x < self.min_units:
            return 0
        if x <= self.mode_units:
            return 1 + (self.peak - 1) * (x - self.min_units) / (
                self.mode_units - self.min_units
            )
        if x <= self.max_units:
            return self.peak - (self.peak - 1) * (x - self.mode_units) / (
                self.max_units - self.mode_units
            )
        return 1

    def proposal_distribution(self, x):
        return self.rng.uniform()

    def sample(self, num_units):
        if num_units < self.min_units:
            # log.warning(f"Requested {num_units} units, but minimum is {self.min_units}")
            return num_units

        # uses rejection sampling
        num_attempts = 0
        while True:
            x = self.min_units + self.rng.uniform() * (
                self.max_units - self.min_units
            )  # Sample from the proposal distribution
            u = self.rng.uniform()
            if u <= self.unnormalized_density_function(x) / (
                self.M * self.proposal_distribution(x)
            ):
                return x
            num_attempts += 1
            if num_attempts > self.max_attempts:
                # warning
                log.warning(
                    f"Could not sample from distribution after {num_attempts} attempts, using all units"
                )
                return num_units


class UnitDropout:
    r"""Augmentation that randomly drops units from the data object.
    
    ..note:: 
        Use `UnitDropout` before `RandomCrop` would be more efficient if index_maps are 
        precomputed. 
    """
    def __init__(self, *args, **kwargs):
        self.distribution = UnitCustomDistribution(*args, **kwargs)

    def __call__(self, data):
        # get units
        unit_ids = data.units.id
        num_units = len(unit_ids)

        num_units_to_sample = int(self.distribution.sample(num_units))

        # shuffle units and take the first num_units_to_sample
        # torch.randperm(num_units)[:num_units_to_sample]
        unit_splits = np.random.permutation(num_units)
        drop_indices = np.sort(unit_splits[num_units_to_sample:])

        unit_mask = np.ones_like(unit_ids, dtype=bool)
        unit_mask[drop_indices] = False

        spike_mask = ~np.isin(data.spikes.unit_index, drop_indices)
        
        data.spikes = data.spikes.select_by_mask(spike_mask)
        data.units = data.units.select_by_mask(unit_mask)

        relabel_map = np.zeros(num_units, dtype=np.long)
        relabel_map[unit_mask] = np.arange(unit_mask.sum())

        data.spikes.unit_index = relabel_map[data.spikes.unit_index]

        return data
