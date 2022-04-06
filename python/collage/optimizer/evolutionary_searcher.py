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
from collage.workloads.torch_workloads import get_network_from_torch
from .op_match_logger import OpMatchLogger
import time
from functools import lru_cache
import os
import logging

import gc

from collage.interface import CollageContext

# the goal ('fitness') function to be maximized
class EvolutionarySearcher:
    def __init__(self, op_state_to_match_translator, expr, net_name, build_target, batch_size,
                 n_ops, match_path, pop_size=100, cx_prob=0.5, mut_prob=0.2, max_iter=100):

        self.op_state_to_match_translator = op_state_to_match_translator
        self.op_match_logger = OpMatchLogger()
        self.n_ops = n_ops
        self.match_path = match_path

        # Debug usage to limit # of measurements
        self.n_test = 0

        # duplicate checker
        # self._memo_state = {}

        self.expr = expr

        # Load network to measure
        self.net_name = net_name
        self.batch_size = batch_size
        self.target_str = build_target

        self.mod, self.params, self.shape_dict, _ = get_network_from_torch(net_name, batch_size)
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
            if best_ind_perf < cur_pop_best_ind_perf:
                best_ind = cur_pop_best_ind

        return best_ind

    # We use subprocess to prevent memory leak;
    # We can remove memory usage of subprocess by discarding subprocess
    # There is no more oom with this!
    def measure_subprocess(self):
        from subprocess import Popen, PIPE, STDOUT, DEVNULL
        from collage.interface import CollageContext

        env = dict(os.environ)
        assert("COLLAGE_HOME" in env)
        script_path = f"{env['COLLAGE_HOME']}/python/collage/testing/tmp_measure_network.py"
        autotvm_tuning_log = CollageContext.pattern_registry.backend_registry["autotvm"].kwargs["tuning_log"]
        backend_list_str = ",".join(CollageContext.backends)
        cmd = [
                'python3',
                script_path,
                self.net_name,
                self.target_str,
                str(self.batch_size),
                autotvm_tuning_log,
                backend_list_str,
                CollageContext.op_cost_logger.log_path,
                CollageContext.op_level_placement_log,
                CollageContext.graph_level_placement_log,
                CollageContext.graph_level_tmp_file,
                str(CollageContext.evolutionary_search_pop_size),
                str(CollageContext.evolutionary_search_max_iter),
                str(CollageContext.evolutionary_search_budget)
            ]

        p = Popen(cmd, stdout=DEVNULL, stderr=PIPE)

        p.wait()
        out, err = p.communicate()

        try:
            res = err.decode("utf-8").partition("##result:")
            numbers = res[2].split()
            mean_perf, std_perf = float(numbers[0]), float(numbers[1])
        except:
            logging.info("Error message from subprocess")
            logging.info(err)
            mean_perf, std_perf = 1000000, 0
            # raise

        return mean_perf, std_perf

    @lru_cache(maxsize=1000)
    def measure_with_lru_cache(self, individual_hash):
        measure_start_time = time.time()
        individual = self.get_individual_from_hash(individual_hash)
        opt_match = self.op_state_to_match_translator.translate(individual)

        # Dump this opt_match in to files so that build pipeline can read it
        # USER_DEFINED_MATCH_LOG
        self.op_match_logger.save(self.expr, opt_match, log_path=self.match_path)

        # Measure entire computation graph with opt_match
        if individual_hash in self.visited:
            mean_perf = self.visited[individual_hash]
        else:
            self.numDup += 1
            # mean_perf, std_perf = measure_end_to_end_user_defined(self.mod["main"], self.params, self.shape_dict,
            #                                                       self.target_str,
            #                                                       self.net_name, self.hw_name, self.batch_size)
            mean_perf, std_perf = self.measure_subprocess()
        # self._memo_state[individual_hash] = -mean_perf

        # Deallocate opt_match
        del opt_match
        logging.info(f"Measurement time : {time.time()-measure_start_time:.2f}s")
        return -mean_perf,
        # return sum(individual),

    def measure_comp_graph(self, individual):
        # Translate individual into match
        logging.info(f"[Evaluation] Individual: {individual}")

        return self.measure_with_lru_cache(self.get_hash_of_individual(individual))

    def log_best_match_and_perf(self, best_ind, cur_pop_best_ind):
        best_ind = self.update_best_individual(best_ind, cur_pop_best_ind)

        # Dump the best match
        best_opt_match = self.op_state_to_match_translator.translate(best_ind[0])
        self.op_match_logger.save(self.expr, best_opt_match, log_path=CollageContext.graph_level_placement_log)


        # This is inference time in ms
        best_perf = -self.get_ind_perf_from_pair(best_ind)
        #with open(best_perf_log_path, "w") as best_output:
        #    best_output.write(f"Best perf: {best_perf}\n")
        #    best_output.write(f"Best match: {best_ind[0]}\n")
        #     best_output.write(f"-> 0 means optimal from first pass and 1 means TensorRT")

        return best_ind, best_opt_match, best_perf

    # Random search; without evolutioinary method for debugging
    def search_test(self, rnd_seed=64):
        g = 0
        best_ind = None
        search_start_time = time.time()
        while g < self.max_iter:
            start_time = time.time()
            g += 1
            logging.info(f"\nGeneration {g} "+ "-" * 30)
            pop = [np.random.randint(2, size=self.n_ops).tolist() for i in range(self.pop_size)]
            if g == 1:
                pop[0] = [0 for i in range(self.n_ops)]
                pop[1] = [1 for i in range(self.n_ops)]

            pop_hash = list(map(self.get_hash_of_individual, pop))
            pop_eval = list(map(self.measure_with_lru_cache, pop_hash))

            logging.info(f"Pop Eval: {pop_eval}")
            max_idx = np.argmax(pop_eval, axis=0)[0]
            cur_pop_best_ind = (pop[max_idx], pop_eval[max_idx])
            best_ind, best_opt_match, best_perf = self.log_best_match_and_perf(best_ind, cur_pop_best_ind)



        logging.info(f"Total search time: {time.time() - search_start_time:.2f}s")

        return best_opt_match

    def save_time_perf_log(self, time_perf_dic, total_search_time, best_perf):
        time_perf_dic[total_search_time] = best_perf
        df = pd.DataFrame.from_dict(time_perf_dic, orient="index")

        # For better printing
        time_perf_log_path = f"{EVAL_RESULT_LOG_PATH}/time_perf_{opt_info_tag}.log"
        df.columns = ["best performance (ms)"]
        df.index.name = "search time (secs)"
        df.to_csv(time_perf_log_path)

    def update_best_ind_and_time_perf(self, best_ind, pop, search_start_time, time_perf_dic):
        cur_pop_best_ind = tools.selBest(pop, 1)[0]
        # cur_pop_best_ind = tools.selWorst(pop, 1)[0]

        cur_pop_best_ind = (cur_pop_best_ind, cur_pop_best_ind.fitness.values)
        best_ind, best_opt_match, best_perf = self.log_best_match_and_perf(best_ind, cur_pop_best_ind)
        logging.info(f"Best individual up to this generation is {best_ind}")

        # Logging search time and best perf so far

        #self.save_time_perf_log(time_perf_dic, total_search_time, best_perf)

        return best_ind, best_opt_match, time_perf_dic

    def search(self, rnd_seed = 64, n_hours = 0.5):
        # Initialize
        search_start_time = time.time()
        random.seed(rnd_seed)

        # For logging search time and best perf
        time_perf_dic = {}

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

        logging.info("Starting evolutionary search")

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        is_first = True
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            if is_first:
                # Log best op-level performance to show the trend
                #self.save_time_perf_log(time_perf_dic, 0, -fit[0])
                is_first = False

        logging.info("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0

        # Begin the evolution
        while g < self.max_iter:
            # A new generation
            g = g + 1
            logging.info("\n-- Generation %i --" % g)
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
                logging.info(f" Evaluation Elapsed time: {eval_end_time - eval_start_time:.2f}s")
                logging.info("  Evaluated %i individuals" % len(invalid_ind))

                # The population is entirely replaced by the offspring
                pop[:] = offspring

            # Gather all the fitnesses in one list and logging.info the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            logging.info("### Current generation statistics")
            logging.info("  Duplication Rate: %.2f" % (self.numDup/self.pop_size))
            logging.info("  Min: %s" % min(fits))
            logging.info("  Max: %s" % max(fits))
            logging.info("  Avg: %s" % mean)
            logging.info("  Std: %s" % std)
            logging.info("")

            # Best will choose individual with the biggest negative inference time
            # Warning(@Soo): Note that best_ind is a pair of individual and its perf (negative inference time)
            best_ind, best_opt_match, time_perf_dic = self.update_best_ind_and_time_perf(best_ind, pop, search_start_time, time_perf_dic)

            total_search_time = time.time() - search_start_time
            # End the program if the time passes;
            # It was 6 before; however, 3 is enough.
            if total_search_time > n_hours * 3600:
                logging.info(f"It exceeds search time limit ({n_hours} hrs), so it stops.")
                break

        logging.info("-- End of (successful) evolution --")

        logging.info(f"Final best individual is {best_ind}")
        # Note that this search time includes time elapsed in the subprocess
        logging.info(f"Total search time: {total_search_time:.2f}s")

        return best_opt_match
