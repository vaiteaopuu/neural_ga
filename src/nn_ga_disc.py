from torch import reshape, argmax, tensor, float32 as tfloat32, optim
from torch import float64 as tfloat64, log as tlog, sign as tsign, exp as texp
from torch.nn import LogSoftmax
from torch.nn.utils import clip_grad_norm_
from torch.multiprocessing import set_sharing_strategy

from numpy import array, identity
from numpy.random import uniform

from .model_disc import Mut_operator


set_sharing_strategy('file_system')
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
                      learning_rate=10**-2, temp=1.0, nb_core=1):
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
    # set_num_threads(nb_core)
    # set_start_method("spawn")
    coef_gain, coef_entropy, coef_kld = parms

    bin_rep = identity(len(spins))
    init_pop = initialization(len_conf, nb_conf, spins)
    mut_op = Mut_operator(len_conf * len(spins), nb_conf, nb_state=len(spins),
                          emb_size=len(spins)*len(spins))
    optimizer = optim.Adam(mut_op.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(mut_op.parameters(), lr=learning_rate)

    # first iteration
    conf_pop, pop_bin = read_population(texp(init_pop), nb_conf, len_conf,
                                        spins, bin_rep)
    prev_fit = compute_fit(conf_pop, j_parm, fitness)
    bfit = prev_fit.max()
    trajectory = []

    for step in range(nb_gen):
        # get the modify log(p)
        mutated_pop, mu_, sig_ = mut_op(texp(init_pop), temp=temp)
        # read the configuration from the log(p)
        new_conf_pop, pop_n_bin = read_population(texp(mutated_pop), nb_conf,
                                                  len_conf, spins, bin_rep)
        # compute the fitness
        cur_fit = compute_fit(new_conf_pop, j_parm, fitness)
        trajectory += [(new_conf_pop, cur_fit)]
        if bfit < prev_fit.max():
            bfit = prev_fit.max()

        mutations = pop_n_bin - (pop_n_bin * pop_bin)
        mut_rate = mutations.sum(1).sum() * (1./nb_conf)

        # kl divergence from the mutation
        kl_pop = texp(mutated_pop) * (mutated_pop-init_pop)
        ent_pop = texp(mutated_pop) * (mutated_pop)
        p_mutation = (kl_pop * mutations).mean(1)
        # p_mutation = (kl_pop).sum(1)

        gain = (cur_fit - prev_fit)

        kld = (0.5*(sig_ + mu_**2 - 2*tlog(sig_) - 1)).sum(-1).mean()

        # increase kl for good mutations and decrease for bad ones
        m_gain = (p_mutation * tsign(gain)).mean()
        nov_gain = (ent_pop.mean(1)).mean()

        loss = coef_gain * m_gain \
            + coef_entropy * nov_gain\
            + coef_kld * kld

        loss.backward()

        clip_grad_norm_(mut_op.parameters(), 2)
        optimizer.step()
        optimizer.zero_grad()

        prev_fit = cur_fit
        init_pop = mutated_pop.detach()
        pop_bin = pop_n_bin.detach()
    return trajectory
