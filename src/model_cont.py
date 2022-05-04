from torch import nn, zeros as tzeros
from torch import unsqueeze, eye, flip, exp as texp, reshape
import torch
from sys import stderr
from matplotlib import pyplot as plt

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class Mut_operator(nn.Module):

    def __init__(self, dim_node, nb_seq, dim_hid=2**8, nb_hid=2):
        super(Mut_operator, self).__init__()

        hidden = dim_hid
        self.latent_dim = dim_hid
        nb_hidden = nb_hid

        self.enc_node = nn.ModuleList([nn.Linear(dim_node, hidden)])

        nb_i = hidden
        for li in range(nb_hidden):
            self.enc_node += [nn.Linear(nb_i, nb_i)]

        self.enc_mu = nn.Linear(nb_i, self.latent_dim)
        self.enc_si = nn.Linear(nb_i, self.latent_dim)

        self.dec_node = nn.ModuleList([nn.Linear(self.latent_dim, nb_i)])

        for li in range(nb_hidden):
            self.dec_node += [nn.Linear(nb_i, nb_i)]
        self.dec_node += [nn.Linear(nb_i, dim_node)]

        # print(self, file=stderr)

    def forward(self, conf_, prev_latent=None, p_rate=1.0, temp=1.):
        # save non modif conf
        init_conf = conf_

        for lin in self.enc_node[:-1]:
            conf_ = lin(conf_)
            conf_ = torch.tanh(conf_)
        conf_ = self.enc_node[-1](conf_)

        mu = self.enc_mu(conf_)
        sig = texp(self.enc_si(conf_))
        eps_ = tzeros(mu.shape).data.normal_(0, p_rate)
        conf_ = mu + sig*eps_

        for lin in self.dec_node[:-1]:
            conf_ = lin(conf_)
            conf_ = torch.tanh(conf_)
        conf_ = self.dec_node[-1](conf_)

        conf_ += init_conf
        return conf_, mu, sig, (mu + sig * eps_).detach()
