"""Implementation of the state-of-the-art TTLC ML algorithms"""
import math
import torch as t
import torch.nn as nn
from scipy import io as spio
import torch.nn.functional as F
from modules import torch_helper as th
from modules.transformer.encoder import Encoder

class rnnTTLC(nn.Module):
    """Class for the RNN TTlC Classifier/Regressor in PyTorch"""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 name='rnn',
                 type='lstm',
                 num_layers=1,
                 embedding_dim=100,
                 dense_activation=None,
                 bool_use_bias=True,
                 output_layer=None,
                 in_classifier=None,
                 out_classifier=None,
                 p_dropout=0.5,
                 bool_dropout=False,
                 bool_batchnorm=False,
                 loss=None,
                 path_load_M=None,
                 num_clusters=None,
                 num_training_data=None,
                 lambda_kmeans=0.1,
                 ):
        super(rnnTTLC, self).__init__()
        # Initialise the name of the RNN
        self.name = name

        # Store the number of neurons in the RNN cells
        self.hidden_dim = hidden_dim

        # Bolean whether to use a bias in any of the layers
        self.bool_use_bias = bool_use_bias

        # Parameters for the classifier part of the network
        if output_layer is None:
            self.output_layer = [nn.Linear,
                                 nn.Linear]
        else:
            self.output_layer = output_layer

        # Define the embedding dimension
        self.embedding_dim = embedding_dim

        # Set the number of neurons for the classifier layers
        if in_classifier is None:
            self.in_classifier = [self.hidden_dim,
                                  self.embedding_dim]
        else:
            self.in_classifier = in_classifier
        if out_classifier is None:
            self.out_classifier = [self.embedding_dim,
                                   17]
        else:
            self.out_classifier = out_classifier

        if dense_activation is None:
            self.dense_activation = nn.Tanh() #Tanhshrink
            # self.dense_activation = nn.ReLU() #Tanhshrink
        else:
            self.dense_activation = dense_activation
        # Boolean whether to use dropout in the dense layers
        self.bool_dropout = bool_dropout
        if self.bool_dropout:
            self.dropout = nn.Dropout(p=p_dropout)
        # Boolean whether we want to use batch normalisation in the classifier
        self.bool_batchnorm = bool_batchnorm

        ## Build the RNN
        if type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=num_layers,
                              bias=self.bool_use_bias,
                              batch_first=True)
        elif type == 'gru':
            self.rnn = nn.GRU(input_size=input_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=num_layers,
                              bias=self.bool_use_bias,
                              batch_first=True)
        else:
            raise Warning("the input type={} is not 'lstm' or 'gru'".format(
                type))

        # Create classifier part of the network
        self.classifier = self.buildClassifier()

        # Some extra spice for the k-means friendly spaces
        if loss == 'k-means-friendly':
            # Initialise some parameters required for the cluster centers and
            # cluster labels
            if num_clusters is None:
                self.num_clusters = 7
            else:
                self.num_clusters = num_clusters

            if num_training_data is None:
                self.num_training_data = 1000
            else:
                self.num_training_data = num_training_data

            # Depending on the activation function, the embedding space might be
            # contrained to a certain range
            if self.dense_activation == 'tanh':
                self.embed_max = 1.0
                self.embed_min = -1.0
            elif self.dense_activation == 'relu':
                self.embed_max = 10.0
                self.embed_min = 0.0
            elif self.dense_activation == 'sigmoid':
                self.embed_max = 1.0
                self.embed_min = 0.0
            else:
                self.embed_max = 10.0
                self.embed_min = -10.0
            # define a few variables we need for the K-means friendly loss
            if path_load_M is not None:
                self.load_cluster_centers(path_load_M)
            else:
                # Initialise the cluster centers
                # Make the matrix of cluster centers not trainable for the
                # k-means-friendly loss:
                # (requires_grad=(loss == 'k-means-continuous')
                self.M = (self.embed_min - self.embed_max) * \
                         t.rand(self.num_clusters,
                                self.embedding_dim,
                                requires_grad=loss == 'k-means-continuous') \
                         + self.embed_max
            if loss == 'k-means-friendly':
                # self.M.requires_grad = False
                # Randomly Initialise the cluster labels
                self.S = t.randint(low=0,
                                   high=self.num_clusters,
                                   size=(self.num_training_data,),
                                   requires_grad=False)

            # We will store the lambda used for k-means here
            self.lambda_kmeans = lambda_kmeans

    def forward(self, x):
        # First, extract features using the RNN
        # First use the RNN to extract the features
        out = self.get_features(x)
        # After the feature extraction, we want to classify the data
        for layer in self.classifier:
            out = layer(out)

        return out

    def get_features(self, x):
        """Function to extract the features"""
        out, _ = self.rnn(x)
        return out[:, -1]

    def buildClassifier(self):
        """Build the classifer layers"""
        classifierLayers = list()
        # Loop through the dense classifier layers adding dropout as required
        i = 0
        for layer, input_dim, output_dim in zip(self.output_layer,
                                                self.in_classifier,
                                                self.out_classifier):
            classifierLayers.append(
                layer(in_features=input_dim,
                      out_features=output_dim,
                      bias=self.bool_use_bias)
            )
            if not i == (len(self.output_layer) - 1):
                classifierLayers.append(self.dense_activation)
            if self.bool_batchnorm:
                classifierLayers.append(nn.BatchNorm1d(num_features=output_dim))
            if self.bool_dropout:
                classifierLayers.append(self.dropout)
            # Increment the counter
            i += 1

        return nn.ModuleList(classifierLayers)

    def load_cluster_centers(self, PATH):
        """Load some pretrained cluster centers"""
        M_tmp = spio.loadmat(PATH)
        self.M = t.tensor(M_tmp['cluster_centers'])

    def save_cluster_centers(self,
                             PATH):
        spio.savemat(PATH, dict(cluster_centers=self.M.cpu().detach().numpy()))


class transformerTTLC(nn.Module):
    """Class for the Gated Transformer TTLC Classifier in PyTorch"""

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 length_timeseries: int,
                 hidden_dim: int,
                 num_classes: int,
                 attention_size: int = None,
                 pe: str = "original",
                 name: str = 'gatedTransformer',
                 embedding_dim: int = 100,
                 dense_activation: bool = None,
                 bool_use_bias: bool  = True,
                 output_layer: list = None,
                 in_classifier: list = None,
                 out_classifier: list = None,
                 p_dropout: float = 0.1,
                 bool_dropout: bool = False,
                 bool_batchnorm: bool = False,
                 loss: str = None,
                 path_load_M: str = None,
                 num_clusters: int = None,
                 num_training_data: int = None,
                 lambda_kmeans: float = 0.1,
                 ):

        super(transformerTTLC, self).__init__()
        # Initialise the name of the transformer
        self.name = name

        # Bolean whether to use a bias in any of the layers
        self.bool_use_bias = bool_use_bias

        # Parameters for the classifier part of the network
        if output_layer is None:
            self.output_layer = [nn.Linear,
                                 nn.Linear]
        else:
            self.output_layer = output_layer

        # Define the embedding dimension
        self.embedding_dim = embedding_dim

        # Set the number of neurons for the classifier layers
        if in_classifier is None:
            self.in_classifier = [self.hidden_dim,
                                  self.embedding_dim]
        else:
            self.in_classifier = in_classifier
        if out_classifier is None:
            self.out_classifier = [self.embedding_dim,
                                   17]
        else:
            self.out_classifier = out_classifier

        if dense_activation is None:
            self.dense_activation = nn.Tanh() #Tanhshrink
        else:
            self.dense_activation = dense_activation
        # Boolean whether to use dropout in the dense layers
        self.bool_dropout = bool_dropout
        if self.bool_dropout:
            self.dropout = nn.Dropout(p=p_dropout)
        # Boolean whether we want to use batch normalisation in the classifier
        self.bool_batchnorm = bool_batchnorm

        # Create classifier part of the network
        self.classifier = self.buildClassifier()

        ##
        # Build the transformer parts (from R. Litzka's BA code)
        # This code was adapted from the Gated Transformer method:
        #       https://github.com/ZZUFaceBookDL/GTN
        ###

        self._d_model = d_model
        self._d_input = d_input
        self._length_timeseries = length_timeseries

        self.layers_encoding_time = nn.ModuleList([Encoder(d_model,
                                                           hidden_dim,
                                                           q,
                                                           v,
                                                           h,
                                                           attention_size=attention_size,
                                                           dropout=p_dropout) for _ in
                                                   range(N)])

        self.layers_encoding_channel = nn.ModuleList([Encoder(d_model,
                                                              hidden_dim,
                                                              q,
                                                              v,
                                                              h,
                                                              attention_size=attention_size,
                                                              dropout=p_dropout) for _ in
                                                      range(N)])

        self.gate = nn.Linear(d_model * length_timeseries + d_model * d_input, 2)
        self.gate_dense = nn.Linear(d_model * (length_timeseries + d_input),
                                    self.embedding_dim)

        self.embedding_time = nn.Linear(d_input, d_model)
        self.embedding_channel = nn.Linear(length_timeseries, d_model)

        pe_functions = {
            'original': th.generate_original_PE,
            'regular': th.generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        self.pe = True

        # Some extra spice for the k-means friendly spaces
        if loss == 'k-means-friendly' or loss == 'k-means-continuous':
            # Initialise some parameters required for the cluster centers and
            # cluster labels
            if num_clusters is None:
                self.num_clusters = 7
            else:
                self.num_clusters = num_clusters

            if num_training_data is None:
                self.num_training_data = 1000
            else:
                self.num_training_data = num_training_data

            # Depending on the activation function, the embedding space might be
            # contrained to a certain range
            if self.dense_activation == 'tanh':
                self.embed_max = 1.0
                self.embed_min = -1.0
            elif self.dense_activation == 'relu':
                self.embed_max = 10.0
                self.embed_min = 0.0
            elif self.dense_activation == 'sigmoid':
                self.embed_max = 1.0
                self.embed_min = 0.0
            else:
                self.embed_max = 10.0
                self.embed_min = -10.0
            # define a few variables we need for the K-means friendly loss
            if path_load_M is not None:
                self.load_cluster_centers(path_load_M)
            else:
                # Initialise the cluster centers
                # Make the matrix of cluster centers not trainable for the
                # k-means-friendly loss:
                # (requires_grad=(loss == 'k-means-continuous')
                self.M = (self.embed_min - self.embed_max) * \
                         t.rand(self.num_clusters,
                                self.embedding_dim,
                                requires_grad=loss == 'k-means-continuous') \
                         + self.embed_max
            if loss == 'k-means-friendly':
                # self.M.requires_grad = False
                # Randomly Initialise the cluster labels
                self.S = t.randint(low=0,
                                   high=self.num_clusters,
                                   size=(self.num_training_data,),
                                   requires_grad=False)

            # We will store the lambda used for k-means here
            self.lambda_kmeans = lambda_kmeans


    def forward(self, x):
        # First, extract features using the two towers of the transformer
        out = self.get_features(x)
        # After the feature extraction, we want to classify the data
        for layer in self.classifier:
            out = layer(out)

        return out

    def get_features(self, x):
        """Function to extract the features"""
        K = x.shape[1]

        # Embedding module
        # encoding = self._embedding(x)
        encoding_time = self.embedding_time(x)

        # Add position encoding
        if self.pe:
            pe = t.ones_like(encoding_time[0])
            position = t.arange(0, self._length_timeseries).unsqueeze(-1)
            temp = t.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = t.exp(temp).unsqueeze(0)
            temp = t.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = t.sin(temp)
            pe[:, 1::2] = t.cos(temp)

            encoding_time.add_(pe)
        elif self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding_time.device)
            encoding_time.add_(positional_encoding)

        # Encoding stack
        for layer_time in self.layers_encoding_time:
            encoding_time = layer_time(encoding_time)

        # channel wise:
        encoding_channel = self.embedding_channel(x.transpose(-1, -2))

        for layer_channel in self.layers_encoding_channel:
            encoding_channel = layer_channel(encoding_channel)

        encoding_time = encoding_time.reshape(encoding_time.shape[0], -1)
        encoding_channel = encoding_channel.reshape(encoding_channel.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(t.cat([encoding_time, encoding_channel], dim=-1)), dim=-1)
        encoding = t.cat([encoding_time * gate[:, 0:1], encoding_channel * gate[:, 1:2]],
                             dim=-1)

        # Add one linear layer to get to the embedding dimension for a fair comparison
        encoding = self.dense_activation(self.gate_dense(encoding))

        return encoding

    def buildClassifier(self):
        """Build the classifer layers"""
        classifierLayers = list()
        # Loop through the dense classifier layers adding dropout as required
        i = 0
        for layer, input_dim, output_dim in zip(self.output_layer,
                                                self.in_classifier,
                                                self.out_classifier):
            classifierLayers.append(
                layer(in_features=input_dim,
                      out_features=output_dim,
                      bias=self.bool_use_bias)
            )
            if not i == (len(self.output_layer) - 1):
                classifierLayers.append(self.dense_activation)
            if self.bool_batchnorm:
                classifierLayers.append(nn.BatchNorm1d(num_features=output_dim))
            if self.bool_dropout:
                classifierLayers.append(self.dropout)
            # Increment the counter
            i += 1

        return nn.ModuleList(classifierLayers)

    def load_cluster_centers(self, PATH):
        """Load some pretrained cluster centers"""
        M_tmp = spio.loadmat(PATH)
        self.M = t.tensor(M_tmp['cluster_centers'])

    def save_cluster_centers(self,
                             PATH):
        spio.savemat(PATH, dict(cluster_centers=self.M.cpu().detach().numpy()))
