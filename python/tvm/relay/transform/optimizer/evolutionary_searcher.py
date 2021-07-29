#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random

from deap import base
from deap import creator
from deap import tools
import pandas as pd
from .custom_fusion_pass import measure_end_to_end_user_defined
from ..backend_operator.utils import *
from workloads.relay_workloads import get_network_from_relay
from workloads.torch_workloads import *

from .op_match_logger import OpMatchLogger
from ..backend_operator.target import BEST_MATCH_LOG, LOG_PATH
import time
from functools import lru_cache

from ..utility.debug_helper import printe

import gc

# the goal ('fitness') function to be maximized
class EvolutionarySearcher:
    def __init__(self, op_state_to_match_translator, expr, net_name, hw_name,
                 n_ops, pop_size=100, cx_prob=0.5, mut_prob=0.2, max_iter=100):

        self.op_state_to_match_translator = op_state_to_match_translator
        self.op_match_logger = OpMatchLogger()
        self.n_ops = n_ops

        # Debug usage to limit # of measurements
        self.n_test = 0

        # duplicate checker
        # self._memo_state = {}

        self.expr = expr

        # Load network to measure
        self.net_name = net_name
        self.hw_name = hw_name
        self.target_str = 'cuda'

        self.mod, self.params, self.shape_dict, _ = get_network_from_torch(net_name, 1)
        # self.mod, self.params = get_network_from_relay(net_name, 1)
        # self.shape_dict = {"data": [1, 64, 56, 56]}

        # Hyperparameters
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.max_iter = max_iter


        self.visited = dict()
        self.numDup = 0

        # Prepare creators
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Attribute generator
        #                      define 'gen_state' to be an attribute ('gene')
        #                      which corresponds to integers sampled uniformly
        #                      from the range [0,1] (i.e. 0 or 1 with equal
        #                      probability)
        self.toolbox = base.Toolbox()
        self.toolbox.register("gen_state", random.randint, 0, 1)

        # Structure initializers
        #                         define 'individual' to be an individual
        #                         consisting of 100 'gen_state' elements ('genes')
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                         self.toolbox.gen_state, n_ops)

        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # ----------
        # Operator registration
        # ----------
        # register the goal / fitness function
        self.toolbox.register("evaluate", self.measure_comp_graph)

        # register the crossover operator
        self.toolbox.register("mate", tools.cxTwoPoint)

        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        # self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("select", tools.selTournament, tournsize=5)
        # self.toolbox.register("select", tools.selBest)

    def get_hash_of_individual(self, individual):
        return "".join((map(str, individual)))

    def get_individual_from_hash(self, individual_hash):
        return [int(i) for i in individual_hash]

    def get_ind_perf_from_pair(self, individual):
        return individual[1][0]

    def update_best_individual(self, best_ind, cur_pop_best_ind):
        if best_ind == None:
            best_ind = cur_pop_best_ind
        else:
            # Note that perf is negative inference time
            best_ind_perf = self.get_ind_perf_from_pair(best_ind)
            cur_pop_best_ind_perf = self.get_ind_perf_from_pair(cur_pop_best_ind)
            # printe("*"*30)
            # printe(best_ind, cur_pop_best_ind)
            # perf is negative inference time; the more the better
            if best_ind_perf < cur_pop_best_ind_perf:
                best_ind = cur_pop_best_ind

        return best_ind

    # We use subprocess to prevent memory leak;
    # We can remove memory usage of subprocess by discarding subprocess
    # There is no more oom with this!
    def measure_subprocess(self):
        from subprocess import Popen, PIPE, STDOUT, DEVNULL
        cmd = ['python3',  'testing/tmp_measure_network.py', self.net_name, self.target_str, self.hw_name]
        p = Popen(cmd, stdout=DEVNULL, stderr=PIPE)
        # p = Popen(cmd)
        p.wait()
        out, err = p.communicate()

        try:
            res = err.decode("utf-8").partition("##result:")
            #assert(len(res)==3)
            numbers = res[2].split()
            mean_perf, std_perf = float(numbers[0]), float(numbers[1])
        except:
            printe("Error message from subprocess")
            printe(err)
            raise

        return mean_perf, std_perf

    @lru_cache(maxsize=1000)
    def measure_with_lru_cache(self, individual_hash):
        measure_start_time = time.time()
        individual = self.get_individual_from_hash(individual_hash)
        opt_match = self.op_state_to_match_translator.translate(individual)
        #printe(f"opt_match: {opt_match}")

        # Dump this opt_match in to files so that build pipeline can read it
        # USER_DEFINED_MATCH_LOG
        match_path = f"{LOG_PATH}/user_defined_match_{self.net_name}_{self.hw_name}.log"
        self.op_match_logger.save(self.expr, opt_match, log_path=match_path)
        #printe(f"[Evaluation] Match log saved")
        # Measure entire computation graph with opt_match

        # Debugging code
        """
        perf_arr = []
        for i in range(10):
            # mean_perf, std_perf = measure_end_to_end_user_defined(self.mod["main"], self.params,
            #                                                       self.shape_dict, self.target_str)

            # Warning(@Sung): USE this function to PREVENT MEMORY LEAK!
            mean_perf, std_perf = self.measure_subprocess()
            printe(f"\t> individual {individual} perf: {mean_perf} ")
            perf_arr.append(mean_perf)
        print(f"[Total perf] (mean, std) = ({np.mean(perf_arr)}, {np.std(perf_arr)}")



        self.n_test += 1
        if self.n_test == 2:
            import sys
            sys.exit(0)
        """

        if individual_hash in self.visited:
            mean_perf = self.visited[individual_hash]
        else:
            self.numDup += 1
            # mean_perf, std_perf = measure_end_to_end_user_defined(self.mod["main"], self.params, self.shape_dict,
            #                                                       self.target_str, self.net_name, self.hw_name)
            mean_perf, std_perf = self.measure_subprocess()
        # self._memo_state[individual_hash] = -mean_perf

        # Deallocate opt_match
        del opt_match
        printe(f"Measurement time : {time.time()-measure_start_time:.2f}s")
        return -mean_perf,
        # return sum(individual),

    def measure_comp_graph(self, individual):
        #printe("measure_comp_graph" + "-"*30)
        # Note that the type of individual is (defined as) list

        # If this individual was measured before, we can skip
        # Warning(@Soo): If it takes up too much memory, we can comment this out
        # individual_hash = self.get_hash_of_individual(individual)
        # if individual_hash in self._memo_state:
        #     printe(f"[Evaluation] Individual({individual}) was measured before ")
        #     return self._memo_state[individual_hash],

        # Translate individual into match
        printe(f"[Evaluation] Individual: {individual}")

        return self.measure_with_lru_cache(self.get_hash_of_individual(individual))

    def log_best_match_and_perf(self, best_ind, cur_pop_best_ind):
        best_ind = self.update_best_individual(best_ind, cur_pop_best_ind)

        # Dump the best match
        best_opt_match = self.op_state_to_match_translator.translate(best_ind[0])
        best_match_log_path = f"{BEST_MATCH_LOG}_{self.net_name}_{self.hw_name}.log"
        self.op_match_logger.save(self.expr, best_opt_match, log_path=best_match_log_path)

        # Dump the best performance with best match
        best_perf_log_path = f"{BEST_MATCH_LOG}_{self.net_name}_{self.hw_name}_perf.log"

        # This is inference time in ms
        best_perf = -self.get_ind_perf_from_pair(best_ind)
        with open(best_perf_log_path, "w") as best_output:
            best_output.write(f"Best perf: {best_perf}\n")
            best_output.write(f"Best match: {best_ind[0]}\n")
            best_output.write(f"-> 0 means optimal from first pass and 1 means TensorRT")

        return best_ind, best_opt_match, best_perf

    # Random search; without evolutioinary method for debugging
    def search_test(self, rnd_seed=64):
        g = 0
        best_ind = None
        search_start_time = time.time()
        while g < self.max_iter:
            start_time = time.time()
            g += 1
            printe(f"\nGeneration {g} "+ "-" * 30)
            pop = [np.random.randint(2, size=self.n_ops).tolist() for i in range(self.pop_size)]
            if g == 1:
                pop[0] = [0 for i in range(self.n_ops)]
                pop[1] = [1 for i in range(self.n_ops)]

            pop_hash = list(map(self.get_hash_of_individual, pop))
            pop_eval = list(map(self.measure_with_lru_cache, pop_hash))

            printe(f"Pop Eval: {pop_eval}")
            max_idx = np.argmax(pop_eval, axis=0)[0]
            cur_pop_best_ind = (pop[max_idx], pop_eval[max_idx])
            # printe(f"Best individual before this generation is {best_ind}")
            # printe(f"Best individual for this generation is {cur_pop_best_ind}")
            best_ind, best_opt_match, best_perf = self.log_best_match_and_perf(best_ind, cur_pop_best_ind)
            #printe(f"Best individual up to this generation is {best_ind}")
            #printe(f"Elapsed time: {time.time() - start_time:.2f}s")


        printe(f"Total search time: {time.time() - search_start_time:.2f}s")
        #printe("-" * 30)

        return best_opt_match

    def save_time_perf_log(self, time_perf_dic, total_search_time, best_perf):
        time_perf_dic[total_search_time] = best_perf
        df = pd.DataFrame.from_dict(time_perf_dic, orient="index")

        # For better printing
        time_perf_log_path = f"{LOG_PATH}/time_perf_{self.net_name}_{self.hw_name}.log"
        df.columns = ["best performance (ms)"]
        df.index.name = "search time (secs)"
        df.to_csv(time_perf_log_path)

    def search(self, rnd_seed = 64):
        # Initialize
        search_start_time = time.time()
        random.seed(rnd_seed)

        # Pair of individual and fitness score (negative inference time)
        best_ind = None
        best_opt_match = None

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = self.toolbox.population(n=self.pop_size)

        # Warning(@Soo): Force initial population to have best results from first level and TensorRT
        assert self.pop_size >= 2
        pop[0] = creator.Individual([0 for i in range(self.n_ops)])
        pop[1] = creator.Individual([1 for i in range(self.n_ops)])

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        # CXPB, MUTPB = 0.5, 0.2

        printe("Starting evolutionary search")

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        printe("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0

        # For logging search time and best perf
        time_perf_dic = {}
        # Begin the evolution
        while g < self.max_iter:
            # A new generation
            g = g + 1
            printe("\n-- Generation %i --" % g)
            self.numDup = 0

            if g > 1:
                # Select the next generation individuals
                offspring = self.toolbox.select(pop, len(pop))

                # Clone the selected individuals
                offspring = list(map(self.toolbox.clone, offspring))

                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):

                    # cross two individuals with probability CXPB
                    if random.random() < self.cx_prob:
                        self.toolbox.mate(child1, child2)

                        # fitness values of the children
                        # must be recalculated later
                        del child1.fitness.values
                        del child2.fitness.values


                for mutant in offspring:

                    # mutate an individual with probability MUTPB
                    if random.random() < self.mut_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                eval_start_time = time.time()
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                eval_end_time = time.time()
                printe(f" Evaluation Elapsed time: {eval_end_time - eval_start_time:.2f}s")
                printe("  Evaluated %i individuals" % len(invalid_ind))

                # The population is entirely replaced by the offspring
                pop[:] = offspring

            # Gather all the fitnesses in one list and printe the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            printe("### Current generation statistics")
            printe("  Duplication Rate: %.2f" % (self.numDup/self.pop_size))
            printe("  Min: %s" % min(fits))
            printe("  Max: %s" % max(fits))
            printe("  Avg: %s" % mean)
            printe("  Std: %s" % std)
            printe("")

            # Best will choose individual with the biggest negative inference time
            # Warning(@Soo): Note that best_ind is a pair of individual and its perf (negative inference time)
            cur_pop_best_ind = tools.selBest(pop, 1)[0]
            # cur_pop_best_ind = tools.selWorst(pop, 1)[0]
            cur_pop_best_ind = (cur_pop_best_ind, cur_pop_best_ind.fitness.values)
            best_ind, best_opt_match, best_perf = self.log_best_match_and_perf(best_ind, cur_pop_best_ind)


            # Deallocate memory for useless space
            # if g < self.max_iter:
            #     del best_opt_match
            #     gc.collect()

            printe(f"Best individual up to this generation is {best_ind}")

            # Logging search time and best perf so far
            total_search_time = time.time() - search_start_time
            self.save_time_perf_log(time_perf_dic, total_search_time, best_perf)

            # End the program if the time passes;
            n_hours = 24
            if total_search_time > n_hours * 3600:
                printe(f"It exceeds search time limit ({n_hours} hrs), so it stops.")
                break

        printe("-- End of (successful) evolution --")

        printe(f"Final best individual is {best_ind}")
        # Note that this search time includes time elapsed in the subprocess
        printe(f"Total search time: {total_search_time:.2f}s")
        # printe(self.op_state_to_match_translator.optimized_match)

        # printe("-"*30)
        # printe(best_opt_match)
        return best_opt_match
