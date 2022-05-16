#!/usr/bin/env python

from .model_cont import Mut_operator

from torch import tensor, optim, float64 as tfloat64
from torch import argmin, zeros, manual_seed, cat as tcat
from torch.nn import Softmax

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
                      seed_v=1, dim_hid=2**8, nb_hid=2, dim_lat=2**4):
    # create the initial population
    manual_seed(seed_v)
    pop = create_init_pop(coor_size, pop_size)
    mut_op = Mut_operator(pop.shape[1], pop_size, dim_hid=dim_hid,
                          nb_hid=nb_hid, dim_lat=dim_lat)
    optimizer = optim.Adam(mut_op.parameters(), lr=lr)

    cur_fit = compute_fit(pop, fitness)
    bfit = cur_fit.min()
    trajectory = [bfit.item()]
    soft = Softmax(dim=-1)

    for step in range(nb_gen):
        pop_n = mut_op(pop)

        new_fit = compute_fit(pop_n, fitness)

        delta_r = pop_n - pop
        max_id = argmin(new_fit)
        min_dir = pop_n[max_id, :].detach()

        loss = (((min_dir - pop_n)**2).sum(1)).mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        pop = pop_n.clone().detach()
        cur_fit = new_fit.detach()

        # all_pop = tcat((pop, pop_n), dim=0)
        # all_fit = tcat((cur_fit, new_fit), dim=0)
        # wei_f = soft(-all_fit)
        # pop_ids = choice(list(range(all_pop.shape[0])), p=wei_f, size=pop_size)
        # pop = all_pop[pop_ids, :].detach()
        # pop = all_pop[pop_ids, :].detach()
        # cur_fit = all_fit[pop_ids].detach()

        if bfit > cur_fit.min():
            bfit = cur_fit.min()

        trajectory += [bfit.item()]
    return trajectory
