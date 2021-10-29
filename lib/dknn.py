'''
Define Deep k-Nearest Neighbor object
'''
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DKNNL2(object):
    """
    An object that we use to create and store a deep k-nearest neighbor (DkNN)
    that uses Euclidean distance as a metric.
    """

    def __init__(self, model, x_train, y_train, x_cal, y_cal, layers, k=75,
                 num_classes=10, ys_train=None, cosine=False, device='cuda'):
        """
        Parameters
        ----------
        model : torch.nn.Module
            neural network model that extracts the representations
        x_train : torch.tensor
            a tensor of training samples with shape (num_train_samples, ) +
            input_shape
        y_train : torch.tensor
            a tensor of labels corresponding to samples in x_train with shape
            (num_train_samples, )
        x_cal : torch.tensor
            a tensor of calibrating samples used to calibrate credibility score
            as described in DkNN paper (Papernot & McDaniel '18)
        y_cal : torch.tensor
            a tensor of labels corresponding to x_cal
        layers : list of str
            a list of layer names that are used in DkNN
        k : int, optional
            the number of neighbors to consider, i.e. k in the kNN part
            (default is 75)
        num_classes : int, optional
            the number of classes (default is 10)
        ys_train : torch.tensor, optional
            specify soft labels for training samples. Must have shape
            (num_train_samples, num_classes). (default is None)
        cosine : bool, optional
            If True, use cosine distance. Else use Euclidean distance.
            (default is False)
        device : str, optional
            name of the device model is on (default is 'cuda')
        """
        self.model = model
        self.cosine = cosine
        self.x_train = x_train
        self.y_train = y_train
        self.ys_train = ys_train
        self.layers = layers
        self.k = k
        self.num_classes = num_classes
        self.device = device
        self.indices = []
        self.activations = {}

        # register hook to get representations
        layer_count = 0
        for name, module in self.model.named_children():
            # if layer name is one of the names specified in self.layers,
            # register a hook to extract the activation at every forward pass
            if name in self.layers:
                module.register_forward_hook(self._get_activation(name))
                layer_count += 1
        assert layer_count == len(layers)
        reps = self.get_activations(x_train, requires_grad=False)

        for layer in layers:
            # build faiss index from the activations by layer
            index = self._build_index(reps[layer].cpu())
            self.indices.append(index)

        # set up calibration for credibility score
        # y_pred = self.classify(x_cal)
        self.A = np.zeros((x_cal.size(0), )) + self.k * len(self.layers)
        # for i, (y_c, y_p) in enumerate(zip(y_cal, y_pred)):
        #     self.A[i] -= y_p[y_c]

    def _get_activation(self, name):
        """Hook used to get activation from specified layer name

        Parameters
        ----------
        name : str
            name of the layer to collect the activations

        Returns
        -------
        hook
            the hook function
        """
        def hook(model, input, output):
            self.activations[name] = output
        return hook

    def _build_index(self, xb):
        """Build faiss index from a given set of samples

        Parameters
        ----------
        xb : torch.tensor
            tensor of samples to build the search index, shape is
            (num_samples, dim)

        Returns
        -------
        index
            faiss index built on the given samples
        """

        d = xb.size(-1)
        # brute-force search on GPU (GPU generally doesn't have enough memory)
        # res = faiss.StandardGpuResources()
        # index = faiss.GpuIndexFlatIP(res, d)

        # brute-force search on CPU
        index = faiss.IndexFlatL2(d)

        index.add(xb.detach().cpu().numpy())
        return index

    def get_activations(self, x, batch_size=500, requires_grad=True,
                        device=None):
        """Get activations at each layer in self.layers

        Parameters
        ----------
        x : torch.tensor
            tensor of input samples, shape = (num_samples, ) + input_shape
        batch_size : int, optional
            batch size (Default is 500)
        requires_grad : bool, optional
            whether or not to require gradients on the activations
            (Default is False)
        device : str
            name of the device the model is on (Default is None)

        Returns
        -------
        activations : dict
            dict of torch.tensor containing activations
        """
        if device is None:
            device = self.device

        # first run through to set an empty tensor of an appropriate size
        with torch.no_grad():
            num_total = x.size(0)
            num_batches = int(np.ceil(num_total / batch_size))
            activations = {}
            self.model(x[0:1].to(device))
            for layer in self.layers:
                size = torch.tensor(self.activations[layer].size()[1:]).prod()
                activations[layer] = torch.empty((num_total, size),
                                                 dtype=torch.float32,
                                                 device=device,
                                                 requires_grad=False)

        with torch.set_grad_enabled(requires_grad):
            for i in range(num_batches):
                begin, end = i * batch_size, (i + 1) * batch_size
                # run a forward pass, the attribute self.activations get set
                # to activations of the current batch
                self.model(x[begin:end].to(device))
                # copy the extracted activations to the dictionary of
                # tensor allocated earlier
                for layer in self.layers:
                    act = self.activations[layer]
                    act = act.view(act.size(0), -1)
                    if self.cosine:
                        act = F.normalize(act, 2, 1)
                    activations[layer][begin:end] = act

            return activations

    def get_neighbors(self, x, k=None, layers=None):
        """Find k neighbors of x at specified layers

        Parameters
        ----------
        x : torch.tensor
            samples to query, shape (num_samples, ) + input_shape
        k : int, optional
            number of neighbors (Default is self.k)
        layers : list of str
            list of layer names to find neighbors on (Default is self.layers)

        Returns
        -------
        output : list
            list of len(layers) tuples of distances and indices of k neighbors
        """
        if k is None:
            k = self.k
        if layers is None:
            layers = self.layers

        output = []
        reps = self.get_activations(x, requires_grad=False)
        for layer, index in zip(self.layers, self.indices):
            if layer in layers:
                rep = reps[layer].detach().cpu().numpy()
                D, I = index.search(rep, k)
                # D, I = search_index_pytorch(index, reps[layer], k)
                # uncomment when using GPU
                # res.syncDefaultStreamCurrentDevice()
                output.append((D, I))
        return output

    def classify(self, x, k=None):
        """Find number of k-nearest neighbors in each class

        Arguments
        ---------
        x : torch.tensor
            samples to query, shape is (num_samples, ) + input_shape
        k : int, optional
            number of neighbors to check (Default is None)

        Returns
        -------
        class_counts : np.array
            array of numbers of neighbors in each class, shape is
            (num_samples, self.num_classes)
        """
        nb = self.get_neighbors(x, k=k)
        class_counts = np.zeros((x.size(0), self.num_classes))
        for (_, I) in nb:
            y_pred = self.y_train.cpu().numpy()[I]
            for i in range(x.size(0)):
                class_counts[i] += np.bincount(
                    y_pred[i], minlength=self.num_classes)
        return class_counts

    def classify_soft(self, x, k=None):
        """Use soft lable for classification

        Arguments
        ---------
        x : torch.tensor
            samples to query, shape is (num_samples, ) + input_shape
        k : int, optional
            number of neighbors to check (Default is None)

        Returns
        -------
        class_counts : np.array
            array of numbers of neighbors in each class, shape is
            (num_samples, self.num_classes)
        """
        nb = self.get_neighbors(x, k=k)
        ys = np.zeros((x.size(0), self.num_classes))
        for (_, I) in nb:
            for i in range(x.size(0)):
                ys[i] += self.ys_train.cpu().numpy()[I[i]].mean(0)
        return ys

    def predict(self, x):
        """Predict label of single sample x"""
        return self.classify(x.unsqueeze(0))[0].argmax()

    # def classify_soft(self, x, layer=None, k=None):
    #     """(Deprecated) Find average of exponential of distance from the query
    #     points to neighbors of each class.
    #
    #     Parameters
    #     ----------
    #     x : torch.tensor
    #         samples to query, shape (num_samples, ) + input_shape
    #     k : int, optional
    #         number of neighbors (Default is self.k)
    #     layers : list of str
    #         list of layer names to find neighbors on (Default is self.layers)
    #
    #     Returns
    #     -------
    #     logits : np.array
    #         array of average of exponential of distance to neighbors in each
    #         class, shape is (num_samples, self.num_classes)
    #     """
    #     temp = 2e-2
    #     if layer is None:
    #         layer = self.layers[-1]
    #     if k is None:
    #         k = self.k
    #     with torch.no_grad():
    #         train_reps = self.get_activations(self.x_train)[layer]
    #         train_reps = train_reps.view(self.x_train.size(0), -1)
    #         reps = self.get_activations(x)[layer]
    #         reps = reps.view(x.size(0), -1)
    #         logits = torch.empty((x.size(0), self.num_classes))
    #         for i, rep in enumerate(reps):
    #             dist = (((rep.view(1, -1) - train_reps)**2).sum(1) / temp).exp()
    #             # cos = ((rep.unsqueeze(0) * train_reps).sum(1) / temp).exp()
    #             # cos = (rep.unsqueeze(0) * train_reps).sum(1)
    #             for label in range(self.num_classes):
    #                 logits[i, label] = dist[self.y_train == label].mean()
    #                 # ind = self.y_train == label
    #                 # logits[i, label] = cos[ind].topk(k)[0].mean()
    #         return logits

    def credibility(self, class_counts):
        """compute credibility of samples given their class_counts"""
        alpha = self.k * len(self.layers) - np.max(class_counts, 1)
        cred = np.zeros_like(alpha)
        for i, a in enumerate(alpha):
            cred[i] = np.sum(self.A >= a)
        return cred / self.A.shape[0]

    def find_nn_diff_class(self, x, label):
        """Find the nearest neighbor of x that has a different class from the
        given label.

        Parameters
        ----------
        x : torch.tensor
            tensor of query samples, shape is (num_samples, ) + input_shape
        label : torch.tensor
            tensor of the labels, shape is (num_samples, )

        Returns
        -------
        nn : np.array
            array of indices of the nearest neighbor of each sample in x that
            has a different label from the one specified
        """
        nn = np.zeros(x.size(0))
        for i in range(x.size(0)):
            found_diff_class = False
            k = 1e2
            # find k nearest neighbors at a time, keep increasing k until at
            # least one sample of a different class is found
            while not found_diff_class:
                _, I = self.get_neighbors(x[i].unsqueeze(0), k=int(k))[0]
                I = I[0]
                ind = np.where(label[i] != self.y_train[I])[0]
                if len(ind) != 0:
                    nn[i] = I[ind[0]]
                    found_diff_class = True
                else:
                    k *= 10

        return nn


# =========================================================================== #


class KNNModel(nn.Module):
    '''
    A Pytorch model that apply an identiy function to the input (i.e. output =
    input). It is used to simulate kNN on the input space so that it is
    compatible with attacks implemented for DkNN.
    '''

    def __init__(self):
        super(KNNModel, self).__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.identity(x)
        return x
