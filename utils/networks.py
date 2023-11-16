import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
This file contains the various network architures developed during this work.
We have:
- HiggsNet, a fully-connected feedforward neural network based on BoostedZNet with some slightly altered architecture
    choices. Hyperparameters will likely differ somewhat to BoostedZNet due to a different design and different physical
    decay processes.
- HiggsGAN, an AC-GAN which is trained and then the discriminator sub-network is used as a classification network like HiggsNet.

'''


'''
-------------------------
        HiggsNet
-------------------------
An easy to use hands on class for varying all aspects of the architecture.
We can easily vary:
- Number of input features
- Number of layers
- Number of hidden nodes
- The activation function
- The number of output classes

We also define weight initialization and the normalization layer within the class. The allows us to deploy this class again in the
HiggsGAN for both the generator and discriminator.
'''
class HiggsNet(nn.Module):
    def __init__(self, input_size, hidden_nodes, num_layers, activation_fn, drop_p, mean_std, num_classes):
        super(HiggsNet, self).__init__()
        self.mean_std = mean_std
        self.drop_p = drop_p

        self.norm = self.norm_layer(input_size)
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()  # Batch Normalization layers

        # First layer
        self.fcs.append(nn.Linear(input_size, hidden_nodes))
        self.bns.append(nn.BatchNorm1d(hidden_nodes))

        # Hidden layers
        if self.drop_p == 0:
            for _ in range(num_layers - 1):
                self.fcs.append(nn.Linear(hidden_nodes, hidden_nodes))
                self.bns.append(nn.BatchNorm1d(hidden_nodes))
        else:
            self.drops = nn.ModuleList()
            self.drops.append(nn.Dropout(p=self.drop_p))
            for _ in range(num_layers - 1):
                self.fcs.append(nn.Linear(hidden_nodes, hidden_nodes))
                self.bns.append(nn.BatchNorm1d(hidden_nodes))
                self.drops.append(nn.Dropout(p=drop_p))

        # Output layer
        self.out = nn.Linear(hidden_nodes, num_classes)

        self.activation_fn = activation_fn
        self.initialize_weights()

    def forward(self, x):
        x = self.norm(x)

        if self.drop_p == 0:
            for fc, bn in zip(self.fcs, self.bns):
                x = self.activation_fn(fc(x))
                x = bn(x)
        else:
            for fc, bn, drop in zip(self.fcs, self.bns, self.drops):
                x = self.activation_fn(fc(x))
                x = bn(x)
                x = drop(x)

        x = self.out(x)
        return x
        '''
        self.norm = self.norm_layer(input_size)
        self.fcs = nn.ModuleList()

        # first layer
        self.fcs.append(nn.Linear(input_size, hidden_nodes))

        # hidden layers
        if self.drop_p == 0:
            for _ in range(num_layers - 1):
                self.fcs.append(nn.Linear(hidden_nodes, hidden_nodes))
        else:
            self.drops = nn.ModuleList()
            self.drops.append(nn.Dropout(p=self.drop_p))
            for _ in range(num_layers - 1):
                self.fcs.append(nn.Linear(hidden_nodes, hidden_nodes))
                self.drops.append(nn.Dropout(p=drop_p))

        # output layer
        self.out = nn.Linear(hidden_nodes, num_classes)

        self.activation_fn = activation_fn
        self.initialize_weights()

    def forward(self, x):
        x = self.norm(x)

        if self.drop_p == 0:
            for fc in self.fcs:
                x = self.activation_fn(fc(x))
        else:
            for fc, drop in zip(self.fcs, self.drops):
                x = self.activation_fn(fc(x))
                x = drop(x)

        x = self.out(x)
        return x
        '''
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m != self.norm:
                if type(self.activation_fn) == nn.ReLU:
                    gain = nn.init.calculate_gain('relu')
                elif type(self.activation_fn) == F.sigmoid:
                    gain = nn.init.calculate_gain('sigmoid')
                else:
                    gain = 1
                nn.init.xavier_normal_(m.weight, gain=gain)

    def norm_layer(self, input_size):
        if self.mean_std is None:
            # If mean_std is None, simply return an identity layer
            return nn.Identity()
        else:
            fc1 = nn.Linear(input_size, input_size)
            mean, std = self.mean_std

            weight_tensor = np.zeros((input_size, input_size))
            bias_tensor = np.zeros(input_size)

            for i, (mean, std) in enumerate(zip(mean, std)):
                weight_tensor[i, i] = 1 / std
                bias_tensor[i] = - mean / std

            fc1.weight = torch.nn.parameter.Parameter(torch.Tensor(weight_tensor))
            fc1.bias = torch.nn.parameter.Parameter(torch.Tensor(bias_tensor))

            fc1.weight.requires_grad = False
            fc1.bias.requires_grad = False

        return fc1
'''
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0, column_idx=None):
        ctx.alpha = alpha
        ctx.column_idx = column_idx
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.column_idx is not None:
            grad_output[:, ctx.column_idx] = grad_output[:, ctx.column_idx].neg() * ctx.alpha
        return grad_output, None, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0, column_idx=None):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha
        self.column_idx = column_idx

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha, self.column_idx)
'''
'''
-------------------------
        HiggsGAN
-------------------------
Due to the flexible nature of the HiggsNet class we can use different instances of the model in the generator and the discriminator.

Generator: takes the training dataset and creates new data from a noise dimension of normally distributed noise.

Discriminator: Composed of two models, a main model that gives feedback to the generator on whether the generator's output is real or fake.
The auxiliary model is trained to classify the output of the generator into the particle decay classes. It is the auxiliary discriminator
that is of interest and becomes our HiggsNet-like classifier for jet tagging.
'''
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_nodes, num_layers, activation_fn, drop_p, num_classes, feature_size):
        super(Generator, self).__init__()

        # generator input will be noise + class label
        self.input_size = noise_dim + num_classes
        self.model = HiggsNet(self.input_size, hidden_nodes, num_layers, activation_fn, drop_p, None, num_classes=feature_size)

    def forward(self, z, labels):
        # Concatenate the noise and the label to produce an input
        x = torch.cat([z, labels], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_nodes, num_layers, activation_fn, drop_p, mean_std, num_classes, cM_ind): #RC_initial_weight):
        super(Discriminator, self).__init__()
        self.cM_ind = cM_ind
        #self.residual_weight_validity = nn.Parameter(torch.tensor(RC_initial_weight,dtype=torch.float32))  # for validity
        #self.residual_weight_label = nn.Parameter(torch.tensor(RC_initial_weight,dtype=torch.float32))  # for label
        self.grl = GradientReversalLayer(alpha=1.0, column_idx=cM_ind)
        # The discriminator will have the main output and an auxiliary classifier
        self.main_model = HiggsNet(input_size, hidden_nodes, num_layers, activation_fn, drop_p, mean_std, 1)
        self.aux_model = HiggsNet(input_size, hidden_nodes, num_layers, activation_fn, drop_p, mean_std, num_classes)  # Using the original mean_std

    def forward(self, x):
        # Get the 'mass' feature from the input
        #mass_feature = x[:, self.cM_ind-1:self.cM_ind]

        x = self.grl(x)

        validity = self.main_model(x)
        # Apply the residual connection for validity
        #adjusted_validity = validity - self.residual_weight_validity * mass_feature

        label = self.aux_model(x)
        # Apply the residual connection for label
        #adjusted_label = label - self.residual_weight_label * mass_feature

        return validity, label


'''
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_nodes, num_layers, activation_fn, drop_p, mean_std, num_classes, cM_ind):
        super(Discriminator, self).__init__()
        self.cM_ind = cM_ind
        # The discriminator will have the main output and an auxiliary classifier
        self.main_model = HiggsNet(input_size, hidden_nodes, num_layers, activation_fn, drop_p, mean_std, 1)
        # Drop the FatElectron_cM mean_std
        # This has been indexed as the 10th column. Do not worry, the dataset class enforces this
        aux_mean_std = [sublist[:self.cM_ind-1] + sublist[self.cM_ind:] for sublist in mean_std]
        self.aux_model = HiggsNet(input_size-1, hidden_nodes, num_layers, activation_fn, drop_p, aux_mean_std, num_classes)

    def forward(self, x):
        validity = self.main_model(x)
        # Drop FatElectron_cM for classification
        x_aux = torch.cat([x[:, :self.cM_ind-1], x[:, self.cM_ind:]], dim=1)
        label = self.aux_model(x_aux)
        return validity, label
'''
'''
-------------------------
       HiggsJetTagger
-------------------------
We have a classifier called HiggsJetTagger, defined using an instance of HiggsNet. This model removes the constrained mass (FatElectron_cM)
and makes classifications based on the other feature columns.
During training the Adversary network takes the output from the tagger and attempts regression to predict the mass value. The MSE loss is computed from
this and combined with the cross entropy loss from the tagger and provided as the loss metric to the tagger.
The Adversary acts as regularization to reduce overfitting to the mass.

'''

class HiggsJetTagger(nn.Module):
    def __init__(self, input_size, hidden_nodes, num_layers, activation_fn, drop_p, mean_std, num_classes, cM_ind):
        super(HiggsJetTagger, self).__init__()
        self.cM_ind = cM_ind
        #aux_mean_std = [sublist[:self.cM_ind-1] + sublist[self.cM_ind:] for sublist in mean_std]
        self.aux_model = HiggsNet(input_size, hidden_nodes, num_layers, activation_fn, drop_p, mean_std, num_classes)

    def forward(self, x):
        #x_aux = torch.cat([x[:, :self.cM_ind-1], x[:, self.cM_ind:]], dim=1)
        label = self.aux_model(x)
        return label

class Adversary(nn.Module):
    def __init__(self, num_classes, hidden_nodes, num_layers, activation_fn, drop_p, mean_std):
        super(Adversary, self).__init__()
        self.model = HiggsNet(num_classes, hidden_nodes, num_layers, activation_fn, drop_p, mean_std=None, num_classes=1)

    def forward(self, x):
        return self.model(x)


'''
-------------------------
    HiggsInvariantMass
-------------------------
'''

# Gradient reversal layer
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class GradientReversalLayer(nn.Module):
    def forward(self, x):
        return GradientReversalFunction.apply(x)

class InvNet(nn.Module):
    def __init__(self, input_size, hidden_nodes, num_layers_feat, num_layers_task, num_layers_mass, activation_fn, drop_p, mean_std, num_classes, num_classes_mass):
        super(InvNet, self).__init__()

        # Feature Extractor (with the given mean and std normalization)
        self.feature_extractor = nn.Sequential(
            HiggsNet(input_size=input_size, hidden_nodes=hidden_nodes, num_layers=num_layers_feat, activation_fn=activation_fn, drop_p=drop_p, mean_std=mean_std, num_classes=hidden_nodes),
            nn.BatchNorm1d(hidden_nodes)
        )

        # Task Classifier
        self.task_classifier = HiggsNet(input_size=hidden_nodes, hidden_nodes=hidden_nodes, num_layers=num_layers_task, activation_fn=activation_fn, drop_p=drop_p, mean_std=None, num_classes=num_classes)

        # Adversary (mass classifier) with Gradient Reversal Layer
        self.mass_classifier = nn.Sequential(
            GradientReversalLayer(),
            HiggsNet(input_size=hidden_nodes, hidden_nodes=hidden_nodes, num_layers=num_layers_mass, activation_fn=activation_fn, drop_p=drop_p, mean_std=None, num_classes=num_classes_mass)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        task_output = self.task_classifier(features)
        mass_output = self.mass_classifier(features)
        return mass_output, task_output
