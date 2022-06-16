import torch
import datagenerator
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.sampler


class StateSpaceDataset(Dataset):
    """An implementation of the abstract base class torch.util.data.Dataset for state space models.

    Arguments:
        X (torch.tensor): (T, dim_x) matrix of training set observations
        Z (torch.tensor, optional): (T, dim_z) matrix of training set latent states
    """

    def __init__(self, X, C, Z=None, Y=None):
        self.X = X
        self.Z = Z
        self.C = C
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, t):
        xt = self.X[t]
        ct = self.C[t]
        if self.Z is not None and self.Y is None:
            zt = self.Z[t]
            return xt, ct, zt
        
        if self.Z is not None and self.Y is not None:
            zt = self.Z[t]
            yt = self.Y[t]
            return xt, ct, zt, yt
        else:
            return xt, ct


class StochasticBatchSampler(torch.utils.data.sampler.Sampler):
    """An implementation of the base class torch.utils.data.sampler.Sampler which can be 
    used as argument for a DataLoader object to handle the sampling procedure during training.
    There are n_batches of size batch_size randomly drawn from the timeseries. This implies that
    usually not every sample of the timeseries is seen in every epoch. To avoid batches with
    fewer samples than batch_size, we are not allowed to draw batch_indices higher than
    (T - batch_size). It is obvious that this procedure tends to draw samples from the middle of
    the timeseries more often than from the two ends.

    Arguments:
        data (torch.tensor): (T, dim_x) matrix of training set observations X_true
        batch_size (int): The number of samples each training batch should contain
    """

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.T = len(self.data)
        if (self.T <= self.batch_size):
            raise ValueError('length of time series must be greater than batch_size')
        self.n_batches = self.T // self.batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_indices = torch.randint(low=0, high=self.T - self.batch_size, size=(self.n_batches,))

        # loop through the n_different batches
        for batch_idx in batch_indices:
            batch = []
            iter_start = batch_idx
            iter_end = iter_start + self.batch_size

            for idx in range(iter_start, iter_end):
                batch.append(idx)

            # by using yield, this function will return a Generator (which is an Iterator).
            yield batch


class ShufflingBatchSampler(torch.utils.data.sampler.Sampler):
    """An implementation of the base class torch.utils.data.sampler.Sampler which can be
    used as argument for a DataLoader object to handle the sampling procedure during training.
    The timeseries of observations is divided into mutually exclusive batches of fixed size, 
    which are then randomly shuffled.

    Arguments:
        data (torch.tensor): (T, dim_x) matrix of training set observations X_true
        batch_size (int): The number of samples each training batch should contain
        drop_last (bool): In case that the length of the timeseries T is not divisible by
                          the batch_size, the last batch will not contain batch_size samples.
                          This boolean variable determines whether to drop that last batch or not.
    """

    def __init__(self, data, batch_size, drop_last=False):
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        if (len(self.data) <= self.batch_size):
            raise ValueError('length of time series must be greater than batch_size')

    def __len__(self):
        if self.drop_last:
            return len(self.data) // self.batch_size
        else:
            return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n_batches = self.__len__()
        # randomly shuffle the (starting) indices of the n_batches.
        batch_indices_shuffled = torch.randperm(n_batches)

        # loop through the n_different batches
        for batch_idx in batch_indices_shuffled:
            batch = []
            iter_start = batch_idx * self.batch_size

            if batch_idx == n_batches - 1 and not self.drop_last:
                iter_end = len(self.data)

            else:
                iter_end = iter_start + self.batch_size

            for idx in range(iter_start, iter_end):
                batch.append(idx)

            # by using yield, this function will return a Generator (which is an Iterator).
            yield batch


if __name__ == '__main__':
    dim_x = 3
    dim_z = 2
    T = 105
    true_model = datagenerator.DataGenerator(dim_x, dim_z)
    X, Z = true_model.generate_timeseries(T)

    a = StateSpaceDataset(X)
    print(a[32])
    b = StateSpaceDataset(X, Z)
    print(b[32])

    BATCH_SIZE = 10
    sampler = ShufflingBatchSampler(X, BATCH_SIZE, False)
    print(sampler)
    print(len(sampler))
    elements = [element for element in sampler]
    print(elements)
    print(len(elements))
    flattened_list = [y for x in elements for y in x]
    print(flattened_list)
    print(len(flattened_list))

    sampler_2 = ShufflingBatchSampler(X, BATCH_SIZE, True)
    print(sampler_2)
    print(len(sampler_2))
    elements_2 = [element for element in sampler_2]
    print(elements_2)
    print(len(elements_2))
    flattened_list_2 = [y for x in elements_2 for y in x]
    print(flattened_list_2)
    print(len(flattened_list_2))

    sampler_3 = StochasticBatchSampler(X, BATCH_SIZE)
    print(sampler_3)
    print(len(sampler_3))
    elements_3 = [element for element in sampler_3]
    print(elements_3)
    print(len(elements_3))
    flattened_list_3 = [y for x in elements_3 for y in x]
    print(flattened_list_3)
    print(len(flattened_list_3))

    T = 10
    BATCH_SIZE = 100
    true_model = datagenerator.DataGenerator(dim_x, dim_z)
    X, Z = true_model.generate_timeseries(T)
    try:
        sampler = ShufflingBatchSampler(X, BATCH_SIZE, False)
    except:
        ValueError
        print('value error thrown and caught')

    try:
        sampler = StochasticBatchSampler(X, BATCH_SIZE)
    except:
        ValueError
        print('value error thrown and caught')
