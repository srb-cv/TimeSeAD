from torch.utils.data import Sampler
import random
import math

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.half = batch_size // 2

        self.minority_class = min(set(labels), key=labels.count)
        self.majority_class = max(set(labels), key=labels.count)

        self.min_idx = [i for i, y in enumerate(labels) if y == self.minority_class]
        self.maj_idx = [i for i, y in enumerate(labels) if y == self.majority_class]

    def __iter__(self):
        # shuffle minority each epoch
        min_idx = self.min_idx.copy()
        random.shuffle(min_idx)

        # number of batches = cover all minority samples
        num_batches = math.ceil(len(min_idx) / self.half)

        for i in range(num_batches):
            # minority: no replacement
            min_batch = min_idx[i*self.half:(i+1)*self.half]

            # if last batch is smaller → pad
            if len(min_batch) < self.half:
                min_batch += random.sample(self.min_idx, self.half - len(min_batch))

            # majority: with replacement
            maj_batch = random.choices(self.maj_idx, k=self.half)

            batch = min_batch + maj_batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return math.ceil(len(self.min_idx) / self.half)