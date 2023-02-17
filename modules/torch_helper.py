import os
import pdb
import numpy as np
import torch
import time
import pandas as pd
from modules import config
from modules import utils as ut
from typing import Optional


#
# functions
#
def nof_trainable_parameters(model):
    """Compute the number of trainable parameters of model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)

def get_activation(activation):
    return torch.nn.ModuleDict([
        ['relu', torch.nn.ReLU()],
        ['lrelu', torch.nn.LeakyReLU()],
        ['prelu', torch.nn.PReLU()],
        ['tanh', torch.nn.Tanh()],
        ['swish', Swish()],
    ])[activation]


def compute_cross_entropy_y_true_y_pred():
    """Compute the torch.nn.CrossEntropy"""
    cross_ent_loss = torch.nn.CrossEntropyLoss()

    def compute_loss(model, data, epoch, step, validate=False):
        x, y_true, _ = data
        return cross_ent_loss(
            model(torch.autograd.Variable(x).to(config.device)),
                  torch.autograd.Variable(y_true).to(config.device))
    return compute_loss


def compute_kmeans_friendly_loss():
    """Compute the torch.nn.CrossEntropy"""
    cross_ent_loss = torch.nn.CrossEntropyLoss()

    def compute_loss(model, data, epoch, step, validate=False):
        x, y_true, inds = data
        # need to calculate the current cluster values here for the given batch
        batch_cluster_means = model.M[model.S[inds], :].to(config.device)
        # Calculate the embeddings
        if not model.lambda_kmeans == 0.0:
            x_embed = model.get_features(x).to(config.device)
            # Loss function from the k-means clustering
            loss_kmeans = torch.pow(x_embed - batch_cluster_means, 2).sum(1)
        else:
            loss_kmeans = torch.zeros(1)
        # Calculate the cross entropy loss
        loss_ce = cross_ent_loss(
                model(torch.autograd.Variable(x).to(config.device)),
                    torch.autograd.Variable(y_true).to(config.device)
        )
        # Calculate the total loss
        # print('CE Loss:{:.2f}, kMeans Loss:{}'.format(loss_ce,
        #                                               loss_kmeans.cpu().detach().numpy()),
        #       end=' ')
        loss_tot = loss_ce + model.lambda_kmeans / 2.0 * loss_kmeans.mean()
        return loss_tot

    return compute_loss


def train(
    model,
    epochs,
    train_loader,
    optimizer,
    compute_loss=compute_cross_entropy_y_true_y_pred(),
    val_loader=None,
    lr_scheduler=None,
    regularization_l1=0.0,
    regularization_l2=0.0,
    callbacks=list(),
    verbose=0,
    eval_acc=True,
):
    """
    Train a torch model.

    Args:
        model: A torch model.
        epochs: The number of epochs to train the model.
        train_loader: A DataLoader for the training data.
        optimizer: A torch optimizer.
        compute_loss: A function which computes the loss. It is called as
            compute_loss(model, data, epoch, step, validate)
            where
                data: is either train_data (validate=False) or val_data (validate=True)
                epoch: is the current epoch
                step: is the current gradient step (batch) index
                validate: is False for train_data and True for val_data.
            By default, the torch.nn.CrossEntropyLoss() between model input and predicted model output is computed.
        val_loader: A DataLoader for the validation data. Default: None, no validation data.
        lr_scheduler: A learning rate scheduler. Default: None, no scheduler.
        regularization_l1: l1 weight regularization. Default: 0.0, no regularization.
        regularization_l2: l2 weight regularization. Default: 0.0, no regularization.
        callbacks: A list of instances of the class Callback.
        verbose: Print progress? Set verbose = 2 to print the loss after every epoch. Default: 0, no printing.

    Returns:
        A dictionary with keys "train_loss_per_batch" and "val_loss_per_epoch" containing the training loss per batch
        (of shape (epochs, steps_per_epoch)) and the validation loss per epoch (with shape (epochs,)), respectively.
        Here, steps_per_epoch is the number of gradient steps performed in one epoch (which usually corresponds to the
        number of batches in the data).
    """
    params = dict()
    params['epoch'] = 0
    params['step'] = 0
    params['train_loss'] = None
    params['val_loss'] = None

    history = {
        'train_loss_per_batch': np.zeros((1, len(train_loader))),
        'train_acc_per_batch': np.zeros((1, len(train_loader))),
        'val_loss_per_epoch': np.zeros(1),
        'val_acc_per_epoch': np.zeros(1),
    }

    stop_training = False

    if verbose > 0:
        print('training network...')
    tic_total = time.time()
    for epoch in range(epochs):
        #
        # epoch begin
        #
        tic_epoch = time.time()

        params['epoch'] = epoch
        params['val_loss'] = None

        if epoch > 0:
            history['train_loss_per_batch'] = np.append(
                history['train_loss_per_batch'], np.zeros((1, len(train_loader))), axis=0)
            history['val_loss_per_epoch'] = np.append(history['val_loss_per_epoch'], 0.0)
            history['train_acc_per_batch'] = np.append(
                history['train_acc_per_batch'], np.zeros((1, len(train_loader))), axis=0)
            history['val_acc_per_epoch'] = np.append(history['val_acc_per_epoch'], 0.0)

        for cb in callbacks:
            cb.on_epoch_begin(model, params)

        if verbose > 1:
            print('ep. {}/{}'.format(epoch+1, epochs), end=' ')
        elif verbose > 0:
            # print progress every approximately 10%
            if epochs >= 10:    # do not compute modulo 0
                if (epoch+1) % (epochs//10) == 0:
                    print('ep. {}/{}'.format(epoch+1, epochs))

        for step, train_data in enumerate(train_loader):
            #
            # batch begin
            #
            params['step'] = step

            for cb in callbacks:
                cb.on_batch_begin(model, params)

            optimizer.zero_grad()

            # forward pass
            train_loss = compute_loss(model, train_data, epoch, step, validate=False)

            # weight regularization
            if regularization_l1 != 0 or regularization_l2 != 0:
                weight_norm_1 = 0.0
                weight_norm_2 = 0.0
                for param in model.parameters():
                    weight_norm_1 += torch.norm(param, p=1)
                    weight_norm_2 += torch.norm(param, p=2)
                train_loss += regularization_l1 * weight_norm_1 + regularization_l2 * weight_norm_2

            # backward pass
            train_loss.backward()
            # train_loss.backward(retain_graph=True)

            # optimizer step
            optimizer.step()

            # collect loss
            history['train_loss_per_batch'][epoch, step] = train_loss.item()

            # caclulate the accuracy and append it to the history
            if eval_acc:
                history['train_acc_per_batch'][epoch, step] = evaluate_acc(model,
                                                                           train_data[0],
                                                                           train_data[1])


            params['train_loss'] = train_loss.item()

            #
            # batch end
            #
            for cb in callbacks:
                cb.on_batch_end(model, params)
        del train_loss

        if verbose > 1:
            print('| l. (ep.): {:.2e}'.format(history['train_loss_per_batch'][epoch, :].mean()), end=' ')
            print('| l. (-1): {:.2e}'.format(history['train_loss_per_batch'][epoch, -1]), end=' ')
            print('| acc. (ep.): {:.2f}'.format(history[
                                                    'train_acc_per_batch'][
                                                epoch, :].mean()*100), end=' ')
            print('| acc. (-1): {:.2f}'.format(history['train_acc_per_batch'][epoch, -1]*100), end=' ')

        if val_loader:
            with torch.no_grad():
                val_loss_sum = 0.0
                for i, val_data in enumerate(val_loader):
                    val_loss_sum += compute_loss(model, val_data, epoch, step, validate=True).item()
                val_loss = val_loss_sum / (i+1)
                history['val_loss_per_epoch'][epoch] = val_loss
                # calculate the accuracy on the validation data
                if eval_acc:
                    history['val_acc_per_epoch'][epoch] = evaluate_acc(model,
                                                                       val_data[0],
                                                                       val_data[1])

        else:
            val_loss = None
        params['val_loss'] = val_loss

        # learning rate scheduler step
        if lr_scheduler:
            lr_scheduler.step()

        toc_epoch = time.time()
        # print loss of epoch
        if verbose > 1:
            print('| val l. (ep.): {:.2e}'.format(history['val_loss_per_epoch'][epoch]), end=' ')
            print('| val acc. (ep.): {:.2f}'.format(history['val_acc_per_epoch'][epoch]*100), end=' ')
            print('| eta:', ut.sec_to_hours((toc_epoch-tic_epoch)*(epochs-(epoch+1))), end=' ')
            print('| time:', ut.sec_to_hours(toc_epoch-tic_epoch))

        #
        # epoch end
        #
        for cb in callbacks:
            # if on_epoch_end returns True, the training is stopped
            # (this is for example useful for early stopping)
            if cb.on_epoch_end(model, params):
                stop_training = True
        if stop_training:
            break

    toc_total = time.time()
    if verbose > 0:
        print(
            'best val l. (ep.): {:.2e} in epoch {}'
            .format(history['val_loss_per_epoch'].min(),
                    history['val_loss_per_epoch'].argmin() + 1)
        )
        if eval_acc:
            print(
                'best val acc. (ep.): {:.2f} in epoch {}'
                .format(history['val_acc_per_epoch'].max()*100,
                        history['val_acc_per_epoch'].argmax() + 1)
            )
        print('Elapsed time train():', ut.sec_to_hours(toc_total-tic_total))

    return history


def evaluate_acc(model,
                 X,
                 y,
                 split_size: int = 500):
    """Function to evaluate the accuracy of a model"""
    # Calculate predictions
    with torch.no_grad():
        y_pred = []
        for x_batch in X.split(split_size=split_size):
            y_pred.append(model(x_batch))
        y_pred = torch.vstack(y_pred)
        acc = (y_pred.argmax(1) == y).sum().item() / y.shape[0]
    return acc


def generate_local_map_mask(chunk_size: int,
                            attention_size: int,
                            device: torch.device = 'cpu') -> torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)
    local_map[i, j] = np.abs(i - j) > attention_size
    return torch.BoolTensor(local_map).to(config.device)


def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 24) -> torch.Tensor:
    """Generate positional encoding with a given period.

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    period:
        Size of the pattern to repeat.
        Default is 24.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE

#
# classes
#
class WeightInitializer:
    """
    Initialize the weights of a torch model using the initializer provided to the constructor.

    Example:
        initializer = WeightInitializer(torch.nn.init.xavier_uniform_)
        model.apply(initializer.initialize_weights)
    """
    def __init__(self, initializer, initializer_args=dict()):
        self.initializer = initializer
        if initializer_args is None:
            self.initializer_args = dict()
        else:
            self.initializer_args = initializer_args

    def initialize_weights(self, m):
        if type(m) == torch.nn.Linear:
            self.initializer(m.weight, **self.initializer_args)
        if type(m) == torch.nn.Conv2d:
            self.initializer(m.weight, **self.initializer_args)
        if type(m) == torch.nn.ConvTranspose2d:
            self.initializer(m.weight, **self.initializer_args)
        if type(m) == torch.nn.Conv1d:
            self.initializer(m.weight, **self.initializer_args)
        if type(m).__name__ == 'Conv1dSeparable':
            self.initializer(m.depthwise.weight, **self.initializer_args)
            self.initializer(m.pointwise.weight, **self.initializer_args)
        if type(m).__name__ == 'Conv1dLocal':
            self.initializer(m.weight, **self.initializer_args)


class Swish(torch.nn.Module):
    """
    Swish beschde.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class DataLoader:
    """
    A torch DataLoader-like object for a set of tensors that can be much faster than the torch DataLoader because
    the torch DataLoader grabs individual indices of the dataset and calls cat (slow).

    Note:
        Found online: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, data, batch_size=32, shuffle=False, device=torch.device('cpu')):
        """
        Args:
            data: A tuple of torch tensors to store. All tensors must have the same length .shape[0].
            batch_size: The batch size to load.
            shuffle: If True, shuffle the data "in-place" whenever an iterator is created out of this object.
            device: A torch device. Default: cpu.
                It is assumed that data has already been moved to device. Device is only used to also move the
                permutation indices in case of shuffle = True.
        """
        assert all(d.shape[0] == data[0].shape[0] for d in data)
        self.data = data

        self.dataset_len = self.data[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        # calculate the number of batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        """Re-shuffle the data whenever the DataLoader is used as an iterator."""
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len).to(self.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(d, 0, indices) for d in self.data)
        else:   # no shuffle
            batch = tuple(d[self.i:self.i+self.batch_size] for d in self.data)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class Dataset(torch.utils.data.Dataset):
    """This is meant to be used in conjunction with a torch DataLoader."""
    def __init__(self, data, astype=np.float32):
        """
        Args:
            data: A tuple of numpy arrays. A torch DataLoader will get a list containing one element of each of the
                arrays. It is assumed that each array contains the same number of points, which is equal to
                data[0].shape[0].
            astype: A torch DataLoader will get the data as type astype. Default: np.float32.
        """
        self.data = data
        self.astype = astype

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, index):
        return list(data[index, :].astype(self.astype) for data in self.data)

#
# callbacks
#
class Callback:
    """Used in conjunction with train(). It makes sense to override at least one of the methods. If on_epoch_end()
    returns True, the training is stopped. Otherwise, none of the methods should have a return value.
    """
    def on_batch_begin(self, model, params):
        """Called at the beginning of the for-loop over all training batches."""
        pass

    def on_batch_end(self, model, params):
        """Called at the end of the for-loop over all training batches."""
        pass

    def on_epoch_begin(self, model, params):
        """Called at the beginning of the for-loop over all training epochs."""
        pass

    def on_epoch_end(self, model, params):
        """Called at the end of the for-loop over all training epochs. If True is returned, the training is stopped."""
        return False


class EarlyStopping(Callback):
    """
    If the validation loss does not reach a new minimum for n_epoch_early_stop epochs, training is stopped.
    """
    def __init__(self, n_epoch_early_stop, verbose=False):
        """
        Args:
            n_epoch_early_stop: If there is no new minimum for this number of epochs, the training is stopped.
            verbose: Print a confirmation if the training is stopped early?

        Note:
            The boolean member variable has_stopped_early is True if an early stop occurred. The variable
            epoch_early_stop shows the epoch in which an early stop occurred.
        """
        self.verbose = verbose
        self.n_epoch_early_stop = n_epoch_early_stop
        self.val_loss_min = np.inf
        self.epochs_no_new_val_loss_min = 0
        self.has_stopped_early = False
        self.epoch_early_stop = 0

    def on_epoch_end(self, model, params):
        if params['val_loss']:    # if validation data exists such that a val_loss can be computed
            if params['val_loss'] < self.val_loss_min:
                # new minimum
                self.val_loss_min = params['val_loss']
                self.epochs_no_new_val_loss_min = 0
            else:
                # no new minimum
                self.epochs_no_new_val_loss_min += 1
            if self.epochs_no_new_val_loss_min >= self.n_epoch_early_stop:
                # create a file to indicate that the training was stopped early
                # open('.early_stop_after_epoch_{}'.format(epoch), 'a').close()
                self.verbose and print('early stop in epoch {}'.format(
                    params['epoch'] + 1))
                self.epoch_early_stop = params['epoch']
                self.has_stopped_early = True
        return self.has_stopped_early


class FeatureClustering(Callback):
    """
    Calculate the feature embedding and clustering metrics every few epochs
    """
    def __init__(self,
                 train_loader,
                 num_epochs=10,
                 num_iters_kmeans=1,
                 num_max_clusters=28,
                 num_iters_kmeans_sklearn=100,
                 num_jobs=8,
                 save_dir=None):
        """
        Args:
            n_epoch_early_stop: If there is no new minimum for this number of epochs, the training is stopped.
            verbose: Print a confirmation if the training is stopped early?

        Note:
            The boolean member variable has_stopped_early is True if an early stop occurred. The variable
            epoch_early_stop shows the epoch in which an early stop occurred.
        """
        # Store the training data to calculate the feature embedding on
        self.train_loader = train_loader
        # Store the results in a pandas dataframe
        columns = ['Model',
                   'kFold',
                   'epoch',
                   'rand_ind',
                   'fowlkes_mallows_ind',
                   'sil_score_Ypred',
                   'sil_score_kMeans',
                   'mutual_info',
                   'num_kmeans_clusters',
                   'loop_kmeans']
        self.df_tmp = pd.DataFrame(columns=columns)
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        # count the number of epochs which have passed
        self.epoch_count = 0
        # Some variables for the k-means
        self.num_iters_kmeans = num_iters_kmeans
        self.num_max_clusters = num_max_clusters
        self.num_iters_kmeans_sklearn = num_iters_kmeans_sklearn
        self.num_jobs = num_jobs

    def on_epoch_end(self, model, params):
        '''We want to calculate the k-means clustering '''
        self.epoch_count += 1
        if self.epoch_count % self.num_epochs == 0:
            # first we must calculate the feature embedding for the current
            # network
            with torch.no_grad():
                # To make sure we don't overload the CUDA memory we will
                # stack the results and then transform the embeddings +
                # labels to a numpy array
                X = list()
                y = list()
                y_pred = list()
                for x_tmp, y_tmp, _ in self.train_loader:
                    X.append(model.get_features(x_tmp))
                    y_pred.append(model(x_tmp).argmax(1))
                    y.append(y_tmp)
                # Detach, move to CPU and convert to a numpy array now
                X = torch.cat(X).cpu().detach().numpy()
                y_pred = torch.cat(y_pred).cpu().detach().numpy()
                y = torch.cat(y).cpu().detach().numpy()
                # X = model.get_features(self.x_in).cpu().detach().numpy()
                # y = model(self.x_in).argmax(1).cpu().detach().numpy()
                print('Training k-means on embeddings...')
            _, _, _, self.df_tmp = \
                ut.perform_kMeans(X_input=X,
                                  y_pred=y_pred,
                                  y_labels=y,
                                  df_tmp=self.df_tmp,
                                  num_iters_kmeans=self.num_iters_kmeans,
                                  num_max_clusters=self.num_max_clusters,
                                  num_iters_kmeans_sklearn=self.num_iters_kmeans_sklearn,
                                  num_jobs=self.num_jobs,
                                  model_name=model.name,
                                  epoch=self.epoch_count
                                     )
            if self.save_dir:
                PATH = os.path.join(self.save_dir,
                                    'cluster_per_epoch.csv')
                self.df_tmp.to_csv(PATH)


class KMeansFriendly(Callback):
    """
    Calculate things required for a k-means friendly space
    """
    def __init__(self,
                 train_loader,
                 num_pretrain_epochs=10,
                 lambda_kmeans=0.001):

        # Store the training data to calculate the feature embedding on
        self.train_loader = train_loader
        self.num_pretrain_epochs = num_pretrain_epochs
        self.lambda_kmeans = lambda_kmeans
        self.epoch_counter = 1

    def on_epoch_begin(self, model, params):
        """On the first epoch we need to calculate the correct clustering"""
        if self.epoch_counter < self.num_pretrain_epochs:
            # self.calculate_current_clusters(model=model)
            model.lambda_kmeans = 0.0
        elif self.epoch_counter == self.num_pretrain_epochs:
            model.lambda_kmeans = self.lambda_kmeans


    def on_epoch_end(self, model, params):
        """We want to calculate the k-means clustering"""
        # increase the epoch counter
        self.epoch_counter += 1
        if not model.lambda_kmeans == 0:
            # First we recalculate the cluster asignments
            self.calculate_current_clusters(model=model)
            # Next we need to update the centroids
            # we will first try just setting the new centroids to be the average
            # of all of the embeddings belonging to that cluster
            self.update_centroids(model=model)

    def calculate_current_clusters(self,
                                   model):
        """Calculate the current cluster labels of the data points in x"""
        with torch.no_grad():
            for x_tmp, _, inds in self.train_loader:
                # Loop through the samples in the batch
                # calculate the feature emeddings
                x = model.get_features(x_tmp).cpu()
                # pdb.set_trace()
                for j in range(x.shape[0]):
                    possible_labels = torch.pow(x[j, :].view((1, -1)) -
                                                model.M, 2).sum(1)
                    # Find the maximum label
                    new_label = torch.argmin(possible_labels)
                    # Store the new labels in the correct indices
                    # torch.index_fill()
                    model.S[inds[j]] = new_label

    def update_centroids(self,
                         model):
        """Calculate some new centroids"""

        with torch.no_grad():
            M_tmp = torch.zeros_like(model.M)
            class_count = torch.zeros(M_tmp.shape[0])
            for x_tmp, _, inds in self.train_loader:
                # first calculate the embeddings
                x = model.get_features(x_tmp).cpu()
                for k in range(M_tmp.shape[0]):
                    class_inds = model.S[inds] == k
                    M_tmp[k, :] += x[class_inds].sum(0)
                    class_count[k] += class_inds.sum()
            # Now we average over the number of elements in each class
            # pdb.set_trace()
            # torch.div(M_tmp, class_count, out=model.M)
            for k in range(M_tmp.shape[0]):
                if class_count[k] == 0:
                    continue
                model.M[k, :] = M_tmp[k, :] / class_count[k]


class InterruptTraining(Callback):
    """If a file named filename exists, the training is stopped at the end of the current epoch."""
    def __init__(self, filename='.interrupt'):
        self.filename = filename

    def on_epoch_end(self, *args):
        if os.path.exists(self.filename):
            return True
        else:
            return False


class History(Callback):
    def __init__(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.history = {
            'train_loss_per_batch': np.zeros((1, steps_per_epoch)),
            'val_loss_per_epoch': np.zeros(1),
        }

    def on_epoch_begin(self, model, params):
        if params['epoch'] > 0:
            self.history['train_loss_per_batch'] = np.append(
                self.history['train_loss_per_batch'], np.zeros((1, self.steps_per_epoch)), axis=0)
            self.history['val_loss_per_epoch'] = np.append(self.history['val_loss_per_epoch'], 0.0)

    def on_batch_end(self, model, params):
        self.history['train_loss_per_batch'][params['epoch'], params['step']] = params['train_loss']

    def on_epoch_end(self, model, params):
        self.history['val_loss_per_epoch'][params['epoch']] = params['val_loss']
        return False

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.history, f)

    def plot(self, filename):
        from matplotlib import pyplot as plt

        x_axis = np.arange(1, len(self.history['val_loss_per_epoch'])+1)
        plt.plot(x_axis, self.history['train_loss_per_batch'][:, -1])
        plt.plot(x_axis, self.history['val_loss_per_epoch'])
        plt.plot(x_axis, self.history['train_loss_per_batch'].mean(axis=1))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(('loss', 'val_loss', 'loss_mean'))
        plt.savefig(filename)
        plt.close()


#
# Custom layers
#
class Conv1dSeparable(torch.nn.Module):
    """"Custom Depthwise Separable 1D convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(Conv1dSeparable, self).__init__()
        self.depthwise = torch.nn.Conv1d(in_channels=in_channels,
                                         out_channels=in_channels,
                                         kernel_size=kernel_size,
                                         bias=bias,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         bias=bias)

    def forward(self, x):
        """Forward function for the new module"""
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Conv1dLocal(torch.nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size,
                 stride=1, bias=False):
        super(Conv1dLocal, self).__init__()
        # output_size = _pair(output_size)
        self.weight = torch.nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size, kernel_size)
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.randn(1, out_channels, output_size)
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        kw = self.kernel_size
        dw = self.stride
        x = x.unfold(2, kw, dw)
        # No need to reshape the tensor when using a 1D conv.
        # x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out
