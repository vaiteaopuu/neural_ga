from torch import nn, zeros as tzeros
from torch import unsqueeze, eye, flip, exp as texp, reshape
import torch
from sys import stderr
from matplotlib import pyplot as plt

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class Mut_operator(nn.Module):

    def __init__(self, dim_node, nb_seq, dim_hid=2**8, nb_hid=2, dim_lat=2**4,
                 scaling=10**-3):
        super(Mut_operator, self).__init__()
        self.scaling = scaling

        hidden = dim_hid
        latent_dim = dim_lat
        nb_hidden = nb_hid

        self.enc_node = nn.ModuleList([nn.Linear(dim_node, hidden)])
        nb_i = hidden
        for li in range(nb_hidden-1):
            self.enc_node += [nn.Linear(hidden, hidden)]
        self.enc_node += [nn.Linear(hidden, latent_dim)]
        self.dec_node = nn.ModuleList([nn.Linear(latent_dim, nb_i)])
        for li in range(nb_hidden-1):
            self.dec_node += [nn.Linear(hidden, hidden)]
        self.dec_node += [nn.Linear(hidden, dim_node)]

    def forward(self, conf_, prev_lat=None):
        # save non modif conf
        init_conf = conf_

        for lin in self.enc_node[:-1]:
            conf_ = lin(conf_)
            conf_ = torch.tanh(conf_)
        conf_ = self.enc_node[-1](conf_)
        eps_ = tzeros(conf_.shape).data.normal_(0, 1.0)
        conf_ += eps_
        if prev_lat is not None:
            conf_ += prev_lat

        for lin in self.dec_node[:-1]:
            conf_ = lin(conf_)
            conf_ = torch.tanh(conf_)
        conf_ = self.dec_node[-1](conf_)

        conf_ = init_conf + conf_ * self.scaling
        return conf_
