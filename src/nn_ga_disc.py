from torch import reshape, argmax, tensor, float32 as tfloat32, optim, clamp, ones as tones
from torch import float64 as tfloat64, log as tlog, sign as tsign, exp as texp
from torch.nn import Softmax, LogSoftmax
from torch.nn.utils import clip_grad_norm_
from torch.multiprocessing import set_sharing_strategy

from numpy import array, identity
from numpy.random import uniform, choice

from .model_disc import Mut_operator


set_sharing_strategy('file_system')
SOFT = Softmax(-1)
LSOFT = LogSoftmax(-1)


def initialization(len_conf, nb_conf, spins):
    "generate the initial population of configurations"
    def rand_seq():
        return [uniform(-1, 1, size=len(spins))/1. for _ in range(len_conf)]
    return LSOFT(tensor(array([rand_seq() for i in range(nb_conf)]),
                        dtype=tfloat32)).view((nb_conf, -1))


def read_population(conf_dist, nb_conf, len_conf, spins, bin_rep):
    "distribution of configuration"
    conf_dist = reshape(conf_dist, (nb_conf, len_conf, len(spins))).detach()
    max_prob_idx = argmax(conf_dist, -1)
    new_bin, new_population = [], []
    for conf_num in max_prob_idx:
        samp_conf = tensor([spins[pi] for pi in conf_num])

        new_population += [samp_conf]
        new_bin += [[bin_rep[pi, :] for pi in conf_num]]
    return new_population, tensor(new_bin).view((nb_conf, -1))


def compute_fit(population, j_parm, fitness):
    uniq_conf = list(set([s for s in population]))
    fit_l_uni = map(fitness, ((conf, j_parm) for conf in uniq_conf))
    fit_l_dic = {seq: fit for seq, fit in zip(uniq_conf, fit_l_uni)}
    fit_l = [fit_l_dic[conf] for conf in population]
    return tensor(fit_l, dtype=tfloat64)


def evolve_population(len_conf, nb_gen, nb_conf, parms, j_parm, spins, fitness,
                      learning_rate=10**-2, dim_hid=2**8, nb_hid=2, dim_lat=2**4):
    """evolve the population with random mutation
    Sample with replacement sequences according to the population fitness
    len_conf = length of the configuration/rna sequence...
    nb_gen   = number of generation
    nb_conf  = population size
    j_parm   = additional parameters to compute the fitness function (target structure...)
    spins    = [-1, 1] for spin models; [0, 1...3] for RNA
    temp     = temperature of the decoding
    parms = tuple for [coef_gain, coef_entropy, coef_kld]
    """
    coef_gain, coef_entropy, coef_kld = parms

    bin_rep = identity(len(spins))
    init_pop = initialization(len_conf, nb_conf, spins)
    mut_op = Mut_operator(len_conf * len(spins), nb_conf, nb_state=len(spins),
                          nb_hid=nb_hid, dim_hid=dim_hid, dim_lat=dim_lat)
    optimizer = optim.Adam(mut_op.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(mut_op.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(mut_op.parameters(), lr=learning_rate, momentum=0.9)

    # first iteration
    conf_pop, pop_bin = read_population(texp(init_pop), nb_conf, len_conf,
                                        spins, bin_rep)
    prev_fit = compute_fit(conf_pop, j_parm, fitness)
    bfit = prev_fit.max()
    trajectory = []

    for step in range(nb_gen):
        optimizer.zero_grad()
        # get the modify log(p)
        mutated_pop, mu, sig = mut_op(texp(init_pop))
        mutated_pop = clamp(mutated_pop, min=-5., max=5.)
        # read the configuration from the log(p)
        new_conf_pop, pop_n_bin = read_population(texp(mutated_pop), nb_conf,
                                                len_conf, spins, bin_rep)
        mutations = pop_n_bin - (pop_n_bin * pop_bin)
        avg_mut = mutations.sum(-1).mean().item()

        # compute the fitness
        cur_fit = compute_fit(new_conf_pop, j_parm, fitness)
        trajectory += [(new_conf_pop, cur_fit)]

        if bfit > prev_fit.min():
            bfit = prev_fit.min()


        # kl divergence from the mutation
        kl_pop = texp(mutated_pop) * (mutated_pop-init_pop)
        ent_pop = texp(mutated_pop) * (mutated_pop)
        p_mutation = (kl_pop * mutations).mean(1)
        not_mutated = (tones(mutations.shape) - mutations) * ent_pop

        gain = (cur_fit - prev_fit)

        # increase kl for good mutations and decrease for bad ones
        # m_gain = (p_mutation.sum(-1) * SOFT(-cur_fit)).mean()
        m_gain = (p_mutation * tsign(gain)).mean()
        nov_gain = (ent_pop.sum(-1)).mean()
        # nov_gain = (ent_pop.sum(1) * SOFT(-cur_fit)).mean()
        kld = (0.5*(sig + mu**2 - 2*tlog(sig) - 1)).sum(-1).mean()

        loss = coef_gain * m_gain \
            + coef_entropy * nov_gain\
            + coef_kld * kld

        # loss = coef_gain * m_gain + coef_entropy * ent_pop.mean(1).mean()
        # loss = coef_gain * nov_gain

        # print("{:5d} {:8.1f} {:8.1f} {:8.1f} {:8.1f}".format(step, cur_fit.min().item(), avg_mut, loss.item(), mutated_pop.sum(-1).mean().item()))
        loss.backward()
        optimizer.step()

        prev_fit = cur_fit.detach()
        init_pop = mutated_pop.detach()
        pop_bin = pop_n_bin.detach()

        # prob = SOFT(-cur_fit/0.1)
        # pop_ids = choice(list(range(nb_conf)), p=prob, size=nb_conf)
        # init_pop = mutated_pop[pop_ids, :].detach()
        # pop_bin = pop_n_bin[pop_ids, :].detach()
        # prev_fit = cur_fit[pop_ids].detach()

    return trajectory
