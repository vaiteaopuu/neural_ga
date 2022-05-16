from torch import nn, zeros as tzeros
from torch import unsqueeze, exp as texp, reshape, log as tlog
import torch
from sys import stderr

torch.autograd.set_detect_anomaly(True)

class Mut_operator(nn.Module):

    def __init__(self, dim_node, nb_seq, nb_state=4, nb_hid=2, dim_hid=256,
                 dim_lat=10):
        super(Mut_operator, self).__init__()
        latent_dim = dim_lat
        self.nb_state = nb_state

        self.enc_node = nn.ModuleList([nn.Linear(dim_node, dim_hid)])
        for li in range(nb_hid-1):
            self.enc_node += [nn.Linear(dim_hid, dim_hid)]
        self.enc_node += [nn.Linear(dim_hid, latent_dim)]
        self.enc_si = nn.Linear(dim_hid, latent_dim)

        self.dec_node = nn.ModuleList([nn.Linear(latent_dim, dim_hid)])
        for li in range(nb_hid-1):
            self.dec_node += [nn.Linear(dim_hid, dim_hid)]
        self.dec_node += [nn.Linear(dim_hid, dim_node)]

        # print(self, file=stderr)

    def forward(self, conf):
        init_msa = conf

        # encode
        for lin in self.enc_node[:-1]:
            conf = lin(conf)
            conf = torch.tanh(conf)
        mu = self.enc_node[-1](conf)
        sig = texp(self.enc_si(conf))
        eps = tzeros(mu.shape).data.normal_(0, 1.)
        # conf += tzeros(conf.shape).data.normal_(0, 1.)
        conf = mu + sig * eps

        # decode
        for lin in self.dec_node[:-1]:
            conf = lin(conf)
            conf = torch.tanh(conf)
        conf = self.dec_node[-1](conf)

        # recurrent link
        conf = tlog(init_msa) + conf
        # conf = init_msa + conf

        fixed_shape = tuple(conf.shape[0:-1])
        conf = unsqueeze(conf, -1)
        conf = conf.view(fixed_shape + (-1, self.nb_state))

        log_p = nn.functional.log_softmax(conf, dim=-1)
        log_p = log_p.view(fixed_shape + (-1,))
        return log_p, mu, sig
