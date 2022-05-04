#!/usr/bin/env python

from .model_cont import Mut_operator

from torch import tensor, float32 as tfloat32, optim, float64 as tfloat64
from torch import argmax, normal, log as tlog, zeros, manual_seed, device
from torch import cat as tcat, zeros
from torch.nn import Softmax
import torch
from numpy.random import choice

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def create_init_pop(nb_coor, pop_size):
    init_pop = zeros((pop_size, nb_coor)).data.uniform_(-512, 512)
    return init_pop


def compute_fit(pop, fitness):
    fit = []
    for conf in pop:
        fit += [fitness(conf.detach().numpy())]
    return tensor(fit, dtype=tfloat64)


def evolve_population(coor_size, nb_gen, pop_size, fitness, temp=1, lr=10**-4,
                      seed_v=1, dim_hid=2**8, nb_hid=2):
    # create the initial population
    manual_seed(seed_v)
    pop = create_init_pop(coor_size, pop_size)
    mut_op = Mut_operator(pop.shape[1], pop_size, dim_hid=dim_hid,
                          nb_hid=nb_hid)
    optimizer = optim.Adam(mut_op.parameters(), lr=lr)
    soft = Softmax(dim=-1)

    cur_fit = compute_fit(pop, fitness)
    bfit = cur_fit.min()
    trajectory = [bfit]

    for step in range(nb_gen):
        pop_n, mu_, sig_, lat = mut_op(pop)

        new_fit = compute_fit(pop_n, fitness)

        delta_f = (new_fit - cur_fit)
        wei_f = soft(-new_fit/temp)

        bar_dir = ((wei_f * pop_n.T).T).sum(0).detach()

        max_id = argmax(wei_f)
        min_dir = pop_n[max_id, :].detach()

        kld = (0.5*(mu_**2 + mu_**2 - 2*tlog(sig_) - 1)).mean(1)

        loss = (((min_dir - pop_n)**2).sum(1)).mean() + kld.mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        pop = pop_n.clone().detach()
        cur_fit = new_fit.detach()

        if bfit > cur_fit.min():
            bfit = cur_fit.min()

        trajectory += [bfit.item()]
    return trajectory
