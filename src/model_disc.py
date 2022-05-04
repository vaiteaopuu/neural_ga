from torch import nn, zeros as tzeros
from torch import unsqueeze, exp as texp, reshape, log as tlog
import torch
from sys import stderr

torch.autograd.set_detect_anomaly(True)

class Mut_operator(nn.Module):

    def __init__(self, dim_node, nb_seq, nb_state=4, emb_size=16):
        super(Mut_operator, self).__init__()
        self.emb_seq = 10
        self.latent_dim = dim_node//2
        self.nb_state = nb_state

        hidden = dim_node//2
        nb_hidden = 1

        self.conv_node = nn.Sequential(
            nn.Conv1d(nb_seq, self.emb_seq, kernel_size=7, padding=3)
            )

        self.dconv_node = nn.Sequential(
            nn.Conv1d(self.emb_seq, nb_seq, kernel_size=7, padding=3)
            )

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

        print(self, file=stderr)


    def forward(self, msa_, temp=1.):
        init_msa = msa_.clone()
        init_shape = msa_.shape

        # apply conv
        msa_ = reshape(msa_, (1, init_shape[0], init_shape[1]))
        msa_ = self.conv_node(msa_)
        msa_ = reshape(msa_, (self.emb_seq, init_shape[1]))

        # encode
        for lin in self.enc_node[:-1]:
            msa_ = lin(msa_)
            msa_ = torch.tanh(msa_)
        msa_ = self.enc_node[-1](msa_)

        # apply mutations
        mu = self.enc_mu(msa_)
        sig = texp(self.enc_si(msa_))
        eps_ = tzeros(mu.shape).data.normal_(0, 1.)
        msa_ = mu + sig * eps_

        # decode
        for lin in self.dec_node[:-1]:
            msa_ = lin(msa_)
            msa_ = torch.tanh(msa_)
        msa_ = self.dec_node[-1](msa_)

        # apply deconv
        msa_ = reshape(msa_, (1, self.emb_seq, init_shape[1]))
        msa_ = self.dconv_node(msa_)
        msa_ = reshape(msa_, (init_shape[0], init_shape[1]))

        # recurrent link
        msa_ += init_msa

        fixed_shape = tuple(msa_.shape[0:-1])
        msa_ = unsqueeze(msa_, -1)
        msa_ = msa_.view(fixed_shape + (-1, self.nb_state))

        log_p = nn.functional.log_softmax(msa_/temp, dim=-1)
        log_p = log_p.view(fixed_shape + (-1,))
        return log_p, mu, sig
