"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import os, sys
import time
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import numpy as np

from copy import deepcopy
import itertools
import pickle
import json

from taylor_utils.utils import *

METHOD_ENCODING = {0: "Taylor_weight", 1: "Random", 2: "Weight norm", 3: "Weight_abs",
                   6: "Taylor_output", 10: "OBD", 11: "Taylor_gate_SO",
                   22: "Taylor_gate", 23: "Taylor_gate_FG", 30: "BN_weight", 31: "BN_Taylor"}

# Method is encoded as an integer that mapping is shown above.
# Methods map to the paper as follows:
# 0 - Taylor_weight - Conv weight/conv/linear weight with Taylor FO In Table 2 and Table 1
# 1 - Random        - Random
# 2 - Weight norm   - Weight magnitude/ weight
# 3 - Weight_abs    - Not used
# 6 - Taylor_output - Taylor-output as is [27]
# 10- OBD           - OBD
# 11- Taylor_gate_SO- Taylor SO
# 22- Taylor_gate   - Gate after BN in Table 2, Taylor FO in Table 1
# 23- Taylor_gate_FG- uses gradient per example to compute Taylor FO, Taylor FO- FG in Table 1, Gate after BN - FG in Table 2
# 30- BN_weight     - BN scale in Table 2
# 31- BN_Taylor     - BN scale Taylor FO in Table 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PruningConfigReader(object):
    def __init__(self):
        self.pruning_settings = {}
        self.config = None

    def read_config(self, filename):
        # reads .json file and sets values as pruning_settings for pruning

        with open(filename, "r") as f:
            config = json.load(f)

        self.config = config

        self.read_field_value("method", 0)
        self.read_field_value("frequency", 500)
        self.read_field_value("prune_per_iteration", 2)
        self.read_field_value("maximum_pruning_iterations", 10000)
        self.read_field_value("starting_neuron", 0)

        self.read_field_value("fixed_layer", -1)
        # self.read_field_value("use_momentum", False)

        self.read_field_value("pruning_threshold", 100)
        self.read_field_value("start_pruning_after_n_iterations", 0)
        # self.read_field_value("use_momentum", False)
        self.read_field_value("do_iterative_pruning", True)
        self.read_field_value("fixed_criteria", False)
        self.read_field_value("seed", 0)
        self.read_field_value("pruning_momentum", 0.9)
        self.read_field_value("flops_regularization", 0.0)
        self.read_field_value("prune_neurons_max", 1)

        self.read_field_value("group_size", 1)

    def read_field_value(self, key, default):
        param = default
        if key in self.config:
            param = self.config[key]

        self.pruning_settings[key] = param

    def get_parameters(self):
        return self.pruning_settings


class pytorch_pruning(object):
    def __init__(self, parameters, arguments=None, pruning_settings=dict(), tasks=None, per_filter_params_count=None, logs_dir=None, logs_fp=None, log_folder=None):
        def initialize_parameter(object_name, settings, key, def_value):
            '''
            Function check if key is in the settings and sets it, otherwise puts default momentum
            :param object_name: reference to the object instance
            :param settings: dict of settings
            :param def_value: def value for the parameter to be putted into the field if it doesn't work
            :return:
            void
            '''
            value = def_value
            if key in settings.keys():
                value = settings[key]
            setattr(object_name, key, value)
        
        self.args = arguments
        self.logs_fp = logs_fp
        self.logs_dir = logs_dir
        self.tasks = tasks
        self.per_filter_params_count = per_filter_params_count
        # store some statistics
        
        self.min_criteria_value = 1e6
        self.max_criteria_value = 0.0
        self.median_criteria_value = 0.0
        self.neuron_units = 0
        self.all_neuron_units = 0
        self.pruned_neurons = 0
        self.gradient_norm_final = 0.0
        self.flops_regularization = 0.0 #not used in the paper
        self.pruning_iterations_done = 0

        # initialize_parameter(self, pruning_settings, 'use_momentum', False)
        initialize_parameter(self, pruning_settings, 'pruning_momentum', 0.9)
        initialize_parameter(self, pruning_settings, 'flops_regularization', 0.0)
        self.momentum_coeff = self.pruning_momentum
        self.use_momentum = self.pruning_momentum > 0.0

        initialize_parameter(self, pruning_settings, 'prune_per_iteration', 1)
        initialize_parameter(self, pruning_settings, 'start_pruning_after_n_iterations', 0)
        initialize_parameter(self, pruning_settings, 'prune_neurons_max', 0)
        initialize_parameter(self, pruning_settings, 'maximum_pruning_iterations', 0)
        initialize_parameter(self, pruning_settings, 'pruning_silent', False)
        initialize_parameter(self, pruning_settings, 'l2_normalization_per_layer', False)
        initialize_parameter(self, pruning_settings, 'fixed_criteria', False)
        initialize_parameter(self, pruning_settings, 'starting_neuron', 0)
        initialize_parameter(self, pruning_settings, 'frequency', 30)
        initialize_parameter(self, pruning_settings, 'pruning_threshold', 100)
        initialize_parameter(self, pruning_settings, 'fixed_layer', -1)
        initialize_parameter(self, pruning_settings, 'combination_ID', 0)
        initialize_parameter(self, pruning_settings, 'seed', 0)
        initialize_parameter(self, pruning_settings, 'group_size', 1)

        initialize_parameter(self, pruning_settings, 'method', 0)

        # Hessian related parameters
        self.temp_hessian = [] # list to store Hessian
        self.hessian_first_time = True

        self.parameters = list()

        ##get pruning parameters
        for parameter in parameters:
            # print('parameter:', parameter) # Gate Layers
            parameter_value = parameter["parameter"]
            self.parameters.append(parameter_value)

        # print('-'*10, 'self.parameters', '-'*10)
        # print(self.parameters)
        # print('-'*10)

        if self.fixed_layer == -1:
            ##prune all layers
            self.prune_layers = [True for parameter in self.parameters]
        else:
            ##prune only one layer
            self.prune_layers = [False, ]*len(self.parameters)
            self.prune_layers[self.fixed_layer] = True

        self.iterations_done = 0

        # baseline importance criteria
        self.prune_network_accomulate = {"by_layer": list(), "averaged": list(), "averaged_cpu": list()}


        self.prune_network_criteria = list()
        self.pruning_gates = list()
        self.pruning_gates_cosflags = list()

        for layer in range(len(self.parameters)):
            self.prune_network_criteria.append(list())

            for key in self.prune_network_accomulate.keys():
                self.prune_network_accomulate[key].append(list())

            # self.pruning_gates.append(np.ones(len(self.parameters[layer]),))
            self.pruning_gates.append(torch.ones(len(self.parameters[layer])).cuda())
            self.pruning_gates_cosflags.append(torch.ones(len(self.parameters[layer])).cuda())
            
            layer_now_criteria = self.prune_network_criteria[-1]
            for unit in range(len(self.parameters[layer])):
                layer_now_criteria.append(0.0)

        if not isinstance(self.tasks, type(None)):
            # task specific importance criteria
            self.prune_network_accomulate_task = {task: {"by_layer": list(), "averaged": list(), "averaged_cpu": list()} for task in self.tasks}
            for task in self.tasks:
                for key in self.prune_network_accomulate_task[task].keys():
                    for layer in range(len(self.parameters)):
                        self.prune_network_accomulate_task[task][key].append(0.)


            self.prune_network_criteria_task = {task: list() for task in self.tasks}
            for task in self.tasks:
                for layer in range(len(self.parameters)):
                    self.prune_network_criteria_task[task].append(list())
                for layer in range(len(self.parameters)):
                    for unit in range(len(self.parameters[layer])):
                        self.prune_network_criteria_task[task][layer].append(0.)
        else:
            self.prune_network_accomulate_task = None 
            self.prune_network_criteria_task = None 

        self.filter_to_layer = {}
        self.filter_to_gate_unit_index = {}

        self.nfilters = 0
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue
            for unit in range(len(self.parameters[layer])):
                self.filter_to_layer[self.nfilters] = layer 
                self.filter_to_gate_unit_index[self.nfilters] = unit 
                self.nfilters += 1
        
        self.already_pruned = []

        self.layer_grads = {task : list() for task in self.tasks}
        for task in self.tasks:
            for layer in range(len(self.parameters)):
                self.layer_grads[task].append([])

        self.accumulated_cosine_score = [0. for i in range(len(self.parameters))]
        self.average_cosine_score = [0. for i in range(len(self.parameters))] 
        
        self.filterwise_acc_cos_score = [torch.zeros(len(self.parameters[layer])).to(device) for layer in range(len(self.parameters))]
        self.filterwise_average_cos_score = [torch.zeros(len(self.parameters[layer])).to(device) for layer in range(len(self.parameters))]
        
        self.cosine_iterations_done = 0.

        # logging setup
        self.log_folder = log_folder
        # self.folder_to_write_debug = self.log_folder + '/debug/'
        # if not os.path.exists(self.folder_to_write_debug):
        #     os.makedirs(self.folder_to_write_debug)

        self.method_25_first_done = True

        if self.method == 40 or self.method == 50 or self.method == 25:
            self.oracle_dict = {"layer_pruning": -1, "initial_loss": 0.0, "loss_list": list(), "neuron": list(), "iterations": 0}
            self.method_25_first_done = False

        if self.method == 25:
            with open("./utils/study/oracle.pickle","rb") as f:
                oracle_list = pickle.load(f)

            self.oracle_dict["loss_list"] = oracle_list

        self.needs_hessian = False
        if self.method in [10, 11]:
            self.needs_hessian = True

        # useful for storing data of the experiment
        self.data_logger = dict()
        self.data_logger["pruning_neurons"] = list()
        self.data_logger["pruning_accuracy"] = list()
        self.data_logger["pruning_loss"] = list()
        self.data_logger["method"] = self.method
        self.data_logger["prune_per_iteration"] = self.prune_per_iteration
        self.data_logger["combination_ID"] = list()
        self.data_logger["fixed_layer"] = self.fixed_layer
        self.data_logger["frequency"] = self.frequency
        self.data_logger["starting_neuron"] = self.starting_neuron
        self.data_logger["use_momentum"] = self.use_momentum

        self.data_logger["time_stamp"] = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

        if hasattr(self, 'seed'):
            self.data_logger["seed"] = self.seed

        # self.data_logger["filename"] = "%s/data_logger_seed_%d_%s.p"%(log_folder, self.data_logger["seed"], self.data_logger["time_stamp"])
        # if self.method == 50:
        #     self.data_logger["filename"] = "%s/data_logger_seed_%d_neuron_%d_%s.p"%(log_folder, self.starting_neuron, self.data_logger["seed"], self.data_logger["time_stamp"])
        # self.log_folder = log_folder

        # the rest of initializations
        self.pruned_neurons = self.starting_neuron

        self.util_loss_tracker = 0.0
        self.util_acc_tracker = 0.0
        self.util_loss_tracker_num = 0.0

        self.loss_tracker_exp = ExpMeter()
        # stores results of the pruning, 0 - unsuccessful, 1 - successful
        self.res_pruning = 0

        self.iter_step = 0

        self.train_writer = None

        self.set_moment_zero = True
        self.pruning_mask_from = ""

    def _count_number_of_neurons(self):
        '''
        Function computes number of total neurons and number of active neurons
        :return:
        all_neuron_units - number of neurons considered for pruning
        neuron_units     - number of not pruned neurons in the model
        '''

        all_neuron_units = 0
        neuron_units = 0
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            all_neuron_units += len( self.parameters[layer] )
            for unit in range(len( self.parameters[layer] )):
                if len(self.parameters[layer].data.size()) > 1:
                    statistics = self.parameters[layer].data[unit].abs().sum()
                else:
                    statistics = self.parameters[layer].data[unit]
                if statistics > 0.0:
                    neuron_units += 1

        return all_neuron_units, neuron_units
    
    def _get_pruned_s_config(self):

        pruned_s_config = []
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue
            layer_neurons = 0
            for unit in range(len( self.parameters[layer] )):
                if len(self.parameters[layer].data.size()) > 1:
                    statistics = self.parameters[layer].data[unit].abs().sum()
                else:
                    statistics = self.parameters[layer].data[unit]
                if statistics > 0.0:
                    layer_neurons += 1
            pruned_s_config.append(layer_neurons)
        return pruned_s_config
    
    def _get_average_cosine_scores(self):
        return self.average_cosine_score.copy()

    def enforce_pruning(self):
        '''
        Method sets parameters ang gates to 0 for pruned neurons.
        Helpful if optimizer will change weights from being zero (due to regularization etc.)
        '''
        # print('enforcing pruning')

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            for unit in range(len(self.parameters[layer])):
                if self.pruning_gates[layer][unit] == 0.0:
                    self.parameters[layer].data[unit] *= 0.0

    def reset_cosine_accumulated_scores(self):
        for layer in range(len(self.parameters)):
            self.accumulated_cosine_score[layer] = 0.
            self.filterwise_acc_cos_score[layer] = torch.zeros_like(self.filterwise_acc_cos_score[layer])
        self.cosine_iterations_done = 0.

    def add_cosine_criteria(self, objectives, model, optimizer):
        '''
        This method adds criteria to global list given batch stats.
        '''
        # only for vgg16 without BN
        backbone_non_gates = ['0', '3', '7', '10', '14', '17', '20', '24', '27', '30', '34', '37', '40']
        backbone_mod_names = [f'backbone.{m}.weight' for m in backbone_non_gates]   
        layer_counter = {backbone_mod_names[i]: i for i in range(len(backbone_mod_names))}

        tasks = list(objectives.keys())

        total_objective = 0.
        for i, task in enumerate(tasks):
            optimizer.zero_grad()
            obj = objectives[task]
            total_objective += obj
            obj.backward(retain_graph=True) # retaining graph for the successive backward passes for each loss and final backward for combined loss
            
            for name, param in model.named_parameters():
                if name in backbone_mod_names and param.requires_grad:
                    self.layer_grads[task][layer_counter[name]] = torch.clone(param.grad.detach())
        
        total_objective.backward(retain_graph=True)

        for layer, if_prune in enumerate(self.prune_layers):  # Gate Layers
            if not if_prune:
                continue

            nunits = self.parameters[layer].size(0) # #filters in this layer
            eps = 1e-8
            criteria_for_layer = (self.parameters[layer]*self.parameters[layer].grad).detach() # dot product
            criteria_for_layer = criteria_for_layer**2 

            if self.iterations_done == 0:
                self.prune_network_accomulate["by_layer"][layer] = torch.clone(criteria_for_layer)
            else:
                self.prune_network_accomulate['by_layer'][layer] += torch.clone(criteria_for_layer)
        
        for layer in range(len(self.parameters)):
            
            pairwise_cosine_sim = 0.
            pairwise_cosine_sim_per_filter = torch.zeros_like(self.pruning_gates[layer])
            
            for i in range(len(self.tasks)):
                for j in range(i+1, len(self.tasks)):
                    t1 = self.tasks[i]
                    t2 = self.tasks[j]
                
                    nf = self.layer_grads[t1][layer].shape[0]
                    g1 = self.layer_grads[t1][layer].view(nf, -1)
                    g1_norm = torch.linalg.norm(g1, dim=-1) + 1e-30
                    g1 /= g1_norm.view(-1, 1)

                    g2 = self.layer_grads[t2][layer].view(nf, -1)
                    g2_norm = torch.linalg.norm(g2, dim=-1) + 1e-30
                    g2 /= g2_norm.view(-1, 1)

                    dot = (g1*g2).sum(-1) # (#filters,) -> dot product/cosine-sim per filter

                    unpruned_unit_indices = torch.nonzero(self.pruning_gates[layer]).view(-1)
                    unpruned_filters_dot = torch.gather(dot, dim=-1, index=unpruned_unit_indices)

                    average_dot = unpruned_filters_dot.mean()
                    pairwise_cosine_sim += average_dot.item()

                    # pairwise_cosine_sim_per_filter += (dot * self.pruning_gates[layer])
                    for unit in range(len(self.pruning_gates_cosflags[layer])):
                        flag = self.pruning_gates_cosflags[layer][unit]
                        if flag.item() == -1:
                            dot[unit] = -1
                    pairwise_cosine_sim_per_filter += dot 
            
            pairwise_cosine_sim /= (len(self.tasks) * (len(self.tasks)-1) * 0.5)
            self.accumulated_cosine_score[layer] += pairwise_cosine_sim

            pairwise_cosine_sim_per_filter /= (len(self.tasks) * (len(self.tasks)-1) * 0.5)
            # print('pairwise_cosine_sim_per_filter:', pairwise_cosine_sim_per_filter)

            self.filterwise_acc_cos_score[layer] += pairwise_cosine_sim_per_filter
            # print('self.filterwise_acc_cos_score[layer]:', self.filterwise_acc_cos_score[layer])
            # sys.exit()
        # sys.exit()

        self.iterations_done += 1
        self.cosine_iterations_done += 1.
    
    def compute_cosine_saliency(self):
        '''
        Method performs pruning based on precomputed criteria values. Needs to run after add_criteria()
        '''
        
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue
            if self.iterations_done > 0:
                current_layer_scores = self.prune_network_accomulate["by_layer"][layer] / self.iterations_done 

                if self.l2_normalization_per_layer:
                    # print('Doing l2 normalization of scores!')
                    eps = 1e-8
                    current_layer_scores = current_layer_scores / (np.linalg.norm(current_layer_scores) + eps)
            else:
                print("First do some add_criteria iterations")
                exit()

            # make sure that (already) pruned neurons have 0 criteria
            self.prune_network_criteria[layer] = current_layer_scores * self.pruning_gates[layer]

            self.average_cosine_score[layer] = self.accumulated_cosine_score[layer] / self.cosine_iterations_done
            self.filterwise_average_cos_score[layer] = self.filterwise_acc_cos_score[layer] / self.cosine_iterations_done
        
        self.reset_cosine_accumulated_scores()
        
        prune_criteria_flatten = None
        cos_scores_flatten = None
        for layer in range(len(self.prune_network_criteria)):
            if layer == 0:
                prune_criteria_flatten = self.prune_network_criteria[layer].clone()
                cos_scores_flatten = self.filterwise_average_cos_score[layer].clone()
            else:
                prune_criteria_flatten = torch.cat((prune_criteria_flatten, self.prune_network_criteria[layer]), dim=-1)
                cos_scores_flatten = torch.cat((cos_scores_flatten, self.filterwise_average_cos_score[layer]), dim=-1)
        
        # filter_ranks = torch.argsort(prune_criteria_flatten) # !!!!LOWEST TO HIGHEST!!!!

        # count number of neurons
        all_neuron_units, neuron_units = self._count_number_of_neurons()
        self.neuron_units = neuron_units
        self.all_neuron_units = all_neuron_units

        prune_neurons_now = (self.pruned_neurons + self.prune_per_iteration)

        if self.args.cosine_ranking: # sampling based on cosine-scores
            
            print('doing cosing-ranking-prune')

            filter_ranks = torch.argsort(cos_scores_flatten) # !!!!LOWEST TO HIGHEST!!!!

            pruned_till_now = 0.
            counter = 0
            while(counter < len(filter_ranks) and pruned_till_now < self.prune_per_iteration):
                
                f_item = filter_ranks[counter].item()
                layer = self.filter_to_layer[f_item]
                unit_idx = self.filter_to_gate_unit_index[f_item]    
                if self.pruning_gates[layer][unit_idx] == 0: # already pruned
                    counter += 1
                    continue 
                
                remaining_layer_filters = torch.sum(self.pruning_gates[layer]).item()
                if remaining_layer_filters <= 5: # avoid overpruning a layer
                    counter += 1
                    continue

                self.pruning_gates[layer][unit_idx] *= 0.
                self.parameters[layer].data[unit_idx] *= 0.
                self.pruning_gates_cosflags[layer][unit_idx] = -1.
                
                counter += 1
                pruned_till_now += 1

                cos_score = cos_scores_flatten[f_item]                
                # logprint(f'counter: {counter} pruned_till_now: {pruned_till_now} cos_score:{cos_score}', self.logs_fp)

            try:
                self.min_criteria_value = (cos_scores_flatten).min()
                self.max_criteria_value = (cos_scores_flatten).max()
            except:
                self.min_criteria_value = 0.0
                self.max_criteria_value = 0.0

        else: # sampling based on total-pruning 
            print('doing total-ranking-prune')
            filter_ranks = torch.argsort(prune_criteria_flatten) # !!!!LOWEST TO HIGHEST!!!!
            for filter_idx in filter_ranks[:prune_neurons_now]:
                f_item = filter_idx.item()
                layer = self.filter_to_layer[f_item]
                unit_idx = self.filter_to_gate_unit_index[f_item]
                self.pruning_gates[layer][unit_idx] *= 0.
                self.parameters[layer].data[unit_idx] *= 0.
                
                try:
                    self.min_criteria_value = (prune_criteria_flatten).min()
                    self.max_criteria_value = (prune_criteria_flatten).max()
                except:
                    self.min_criteria_value = 0.0
                    self.max_criteria_value = 0.0

        # count number of neurons
        all_neuron_units, neuron_units = self._count_number_of_neurons()

        self.pruning_iterations_done += 1

        self.pruned_neurons = all_neuron_units-neuron_units

        # set result to successful
        self.res_pruning = 1
    
    def do_cosine_step(self, objectives, model, optimizer=None, loss=None, neurons_left=0, training_acc=0.0):
        '''
        do one step of pruning,
        1) Add importance estimate
        2) checks if loss is above threshold
        3) performs one step of pruning if needed
        '''
        self.iter_step += 1
        niter = self.iter_step

        # sets pruned weights to zero
        self.enforce_pruning()

        # compute criteria for given batch
        self.add_cosine_criteria(objectives, model, optimizer)

        # small script to keep track of training loss since the last pruning
        if niter % self.frequency == 0 and niter != 0:
            print('pruning at niter:', niter)

            self.compute_cosine_saliency()

            # training_loss = self.util_training_loss
            if self.res_pruning == 1:
                all_neuron_units, neuron_units = self._count_number_of_neurons()
                # log = f"Pruning: Units: {self.neuron_units}, / {self.all_neuron_units}, loss: {training_loss}, Zeroed: {self.pruned_neurons}, criteria min:{self.min_criteria_value} / max:{self.max_criteria_value:2.7f}"
                log = f"Pruning: Units: {neuron_units}, / {all_neuron_units}, Zeroed: {self.pruned_neurons}, criteria min:{self.min_criteria_value} / max:{self.max_criteria_value:2.7f}"
                logprint(log, self.logs_fp)

    def set_momentum_zero_sgd(self, optimizer=None):
        '''
        Method sets momentum buffer to zero for pruned neurons. Supports SGD only.
        :return:
        void
        '''
        for layer in range(len(self.pruning_gates)):
            if not self.prune_layers[layer]:
                continue
            for unit in range(len(self.pruning_gates[layer])):
                if not self.pruning_gates[layer][unit]:
                    continue
                if 'momentum_buffer' in optimizer.state[self.parameters[layer]].keys():
                    optimizer.state[self.parameters[layer]]['momentum_buffer'][unit] *= 0.0

    def connect_tensorboard(self, tensorboard):
        '''
        Function connects tensorboard to pruning engine
        '''
        self.tensorboard = True
        self.train_writer = tensorboard

    def update_flops(self, stats=None):
        '''
        Function updates flops for potential regularization
        :param stats: a list of flops per parameter
        :return:
        '''
        self.per_layer_flops = list()
        if len(stats["flops"]) < 1:
            return -1
        for pruning_param in self.gates_to_params:
            if isinstance(pruning_param, list):
                # parameter spans many blocks, will aggregate over them
                self.per_layer_flops.append(sum([stats['flops'][a] for a in pruning_param]))
            else:
                self.per_layer_flops.append(stats['flops'][pruning_param])

    def apply_flops_regularization(self, groups, mu=0.1):
        '''
        Function applieregularisation to computed importance per layer
        :param groups: a list of groups organized per layer
        :param mu: regularization coefficient
        :return:
        '''
        if len(self.per_layer_flops) < 1:
            return -1

        for layer_id, layer in enumerate(groups):
            for group in layer:
                # import pdb; pdb.set_trace()
                total_neurons = len(group[0])
                group[1] = group[1] - mu*(self.per_layer_flops[layer_id]*total_neurons)


def prepare_pruning_list(pruning_settings, model, model_name, pruning_mask_from='', name=''):
    '''
    Function returns a list of parameters from model to be considered for pruning.
    Depending on the pruning method and strategy different parameters are selected (conv kernels, BN parameters etc)
    :param pruning_settings:
    :param model:
    :return:
    '''
    # Function creates a list of layer that will be pruned based on user selection

    ADD_BY_GATES = True  # gates add artificially they have weight == 1 and not trained, but gradient is important. see models/lenet.py
    ADD_BY_WEIGHTS = ADD_BY_BN = False

    pruning_method = pruning_settings['method']

    print('pruning_method:', pruning_method)

    pruning_parameters_list = list()
    if ADD_BY_GATES:

        print('adding by gates!')

        first_step = True
        prev_module = None
        prev_module2 = None
        print("network structure")
        for module_indx, m in enumerate(model.modules()):
            # print('module:',module_indx, m)
            if hasattr(m, "do_not_update"):
                m_to_add = m

                print('m_to_add:', m)

                if (pruning_method != 23) and (pruning_method != 6):
                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}
                else:
                    def just_hook(self, grad_input, grad_output):
                        # getting full gradient for parameters
                        # normal backward will provide only averaged gradient per batch
                        # requires to store output of the layer
                        if len(grad_output[0].shape) == 4:
                            self.weight.full_grad = (grad_output[0] * self.output).sum(-1).sum(-1)
                        else:
                            self.weight.full_grad = (grad_output[0] * self.output)

                    if pruning_method == 6:
                        # implement ICLR2017 paper
                        def just_hook(self, grad_input, grad_output):
                            if len(grad_output[0].shape) == 4:
                                self.weight.full_grad_iclr2017 = (grad_output[0] * self.output).abs().mean(-1).mean(
                                    -1).mean(0)
                            else:
                                self.weight.full_grad_iclr2017 = (grad_output[0] * self.output).abs().mean(0)

                    def forward_hook(self, input, output):
                        self.output = output

                    if not len(pruning_mask_from) > 0:
                        # in case mask is precomputed we remove hooks
                        m_to_add.register_forward_hook(forward_hook)
                        m_to_add.register_backward_hook(just_hook)

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                if pruning_method in [30, 31]:
                    # for densenets.
                    # add previous layer's value for batch norm pruning

                    if isinstance(prev_module, nn.BatchNorm2d):
                        m_to_add = prev_module
                        print(m_to_add, "yes")
                    else:
                        print(m_to_add, "no")

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                if pruning_method in [24, ]:
                    # add previous layer's value for batch norm pruning

                    if isinstance(prev_module, nn.Conv2d):
                        m_to_add = prev_module

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                if pruning_method in [0, 2, 3]:
                    # add previous layer's value for batch norm pruning

                    if isinstance(prev_module2, nn.Conv2d):
                        print(module_indx, prev_module2, "yes")
                        m_to_add = prev_module2
                    elif isinstance(prev_module2, nn.Linear):
                        print(module_indx, prev_module2, "yes")
                        m_to_add = prev_module2
                    elif isinstance(prev_module, nn.Conv2d):
                        print(module_indx, prev_module, "yes")
                        m_to_add = prev_module
                    else:
                        print(module_indx, m, "no")

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                pruning_parameters_list.append(for_pruning)
            prev_module2 = prev_module
            prev_module = m

    if model_name == "resnet20":
        # prune only even layers as in Rethinking min norm pruning
        pruning_parameters_list = [d for di, d in enumerate(pruning_parameters_list) if (di % 2 == 1 and di > 0)]

    if ("prune_only_skip_connections" in name) and 1:
        # will prune only skip connections (gates around them). Works with ResNets only
        pruning_parameters_list = pruning_parameters_list[:4]

    print('pruning_parameters_list:', len(pruning_parameters_list))

    return pruning_parameters_list

class ExpMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, mom = 0.9):
        self.reset()
        self.mom = mom

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.exp_avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean_avg = self.sum / self.count
        self.exp_avg = self.mom*self.exp_avg + (1.0 - self.mom)*self.val
        if self.count == 1:
            self.exp_avg = self.val

if __name__ == '__main__':
    pass