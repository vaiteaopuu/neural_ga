from OptimizationTestFunctions import Fletcher, plot_3d, Rastrigin, Eggholder
from OptimizationTestFunctions import Sphere, Ackley, AckleyTest, Rosenbrock, Fletcher, Griewank
from OptimizationTestFunctions import Penalty2, Quartic, Rastrigin, SchwefelDouble, SchwefelMax, SchwefelAbs
from OptimizationTestFunctions import SchwefelSin, Stairs, Abs, Michalewicz, Scheffer, Eggholder, Weierstrass
from scipy import optimize

from src.reg_ga_cont import evolve_population as reg_op
from src.nn_ga_cont import evolve_population as nn_op
import torch
from torch.multiprocessing import Pool

dim = 10

def eval_f(args):
    seed, dim_hid, nb_hid, test_func = args
    nn_traj = nn_op(dim, 1000, 100, test_func, lr=10**-2, seed_v=seed,
                    dim_hid=dim_hid, nb_hid=nb_hid)
    return nn_traj


def write_output(out_res, out_file):
    with open(out_file, "w") as out:
        for val in out_res:
            out.write(f"{val}\n")

def run_de(args):
    test_func, rep = args
    bounds = [(-512, 512)] * dim
    opti_res = optimize.differential_evolution(test_func, bounds,
                                               init="random", popsize=10,
                                               maxiter=1000, polish=False)
    return opti_res, rep, test_func


# BUG: Griewank
all_functions = [Sphere, Ackley, AckleyTest, Rosenbrock, Fletcher, Penalty2,
                 Quartic, Rastrigin, SchwefelDouble, SchwefelMax, SchwefelAbs,
                 SchwefelSin, Stairs, Abs, Michalewicz, Scheffer, Eggholder,
                 Weierstrass]

pool = Pool()
opt_func_l = [opt_func(dim) for opt_func in all_functions]
parms = [(seed, 2**(i+1), nb_hid, test_fun) for i in range(4, 9) for
         nb_hid in range(1, 3) for seed in range(4) for test_fun in opt_func_l]

# print("run opt")
# solus = pool.map(eval_f, parms)
# for nn_traj, (seed, dim_hid, nb_hid, test_func) in zip(solus, parms):
#     write_output(nn_traj, f"data/{type(test_func).__name__}_{dim}_{dim_hid}_{nb_hid}_{seed}.dat")

# all_dim_hid = list(set([dh for _, dh, __, ___ in parms]))


print("run de")
de_solus = pool.map(run_de, [(test_f, rep) for test_f in opt_func_l for rep in
                             range(4)])

for opti_res, rep, test_func in de_solus:
    with open(f"data/{type(test_func).__name__}_{dim}_{rep}.opt", "w") as out_de:
        out_de.write(opti_res.__str__())
