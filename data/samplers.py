"""Training batches that retain every patient without singleton BatchNorm batches."""

from torch.utils.data import BatchSampler


class MergeSingletonBatchSampler(BatchSampler):
    """Append a final singleton to the preceding batch instead of dropping it."""

    def __init__(self, sampler, batch_size):
        if batch_size < 2 or len(sampler) < 2:
            raise ValueError('BatchNorm training requires at least two cases per batch')
        super().__init__(sampler, batch_size, drop_last=False)

    def __iter__(self):
        previous = None
        for batch in super().__iter__():
            if previous is not None:
                if len(batch) == 1:
                    yield previous + batch
                    return
                yield previous
            previous = batch
        if previous is not None:
            yield previous

    def __len__(self):
        count = super().__len__()
        return count - int(len(self.sampler) % self.batch_size == 1)
