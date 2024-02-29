from .unit_dropout import UnitDropout


class Compose:
    r"""Compose several transforms together. All transforms will be called sequentially,
    in order, and must accept and return a single :obj:`poyo.data.Data` object, except
    the last transform, which can return any object.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data