#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/9 11:01
# @Author : Yilu Jiang
# @File : my_genetic.py
import itertools
from abc import ABCMeta, abstractmethod
from time import time
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array

from ._my_program import _Program
from .my_fitness import _fitness_map
from .my_functions import _function_map, _Function, sig1 as sigmoid
from .my_utils import check_random_state, _partition_estimators, unit_transform

__all__ = ['SymbolicRegressor', 'SymbolicClassifier', 'SymbolicTransformer']

MAX_INT = np.iinfo(np.int32).max


def _evolve(n_programs, parents, X, y, seeds, params):
    """Private function used to build a batch of programs."""
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    feature_names = params['feature_names']
    unit_dict = params['unit_dict']

    def _tournament():
        """Find the fittest individual from a subpopulation."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parents_index = contenders[np.argmax(fitness)]
        else:
            parents_index = contenders[np.argmin(fitness)]
        return parents[parents_index], parents_index

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program, random_state)
                genome = {'method': 'Crossover', 'parent_idx': parent_index, 'parent_nodes': removed,
                          'donor_idx': donor_index, 'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation', 'parent_idx': parent_index, 'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation', 'parent_idx': parent_index, 'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation', 'parent_idx': parent_index, 'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction', 'parent_idx': parent_index, 'parent_nodes': []}

        program = _Program(function_set=function_set, arities=arities, init_depth=init_depth, init_method=init_method,
                           n_features=n_features, metric=metric, transformer=transformer, const_range=const_range,
                           p_point_replace=p_point_replace, parsimony_coefficient=parsimony_coefficient,
                           program=program,
                           feature_names=feature_names, unit_dict=unit_dict, random_state=random_state)

        program.parents = genome

        if not program.unit_rationality(X):
            # print(program)
            program.fitness_ = 0
        else:
            program.fitness_ = program.fitness(X, y)
            print(i, program, program.fitness_)
        # if max_samples < n_samples:
        #     # Calculate OOB fitness
        #     indices, not_indices = program.get_all_indices(n_samples, max_samples, random_state)
        #     program.oob_fitness_ = program.fitness(X.iloc[not_indices,:], y.iloc[not_indices,:])

        programs.append(program)

    return programs


class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):
    """Base class for symbolic regression / classification estimators.
       This class should not be used directly.
    """

    @abstractmethod
    def __init__(self, *, population_size=1000, hall_of_fame=None, n_components=None, generations=20,
                 tournament_size=20, stopping_criteria=0.0, const_range=(-1., 1.), init_depth=(2, 6),
                 init_method='half and half', function_set, transformer=None, unit_dict,
                 metric='mean absolute error', parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05, max_samples=1.0, class_weight=None,
                 feature_names=None, warm_start=False, low_memory=False, n_jobs=1, verbose=0, random_state=None):
        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.transformer = transformer
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.unit_dict = unit_dict
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.pre_best = -1

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.
        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average', 'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length', 'Fitness', 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            if self.max_samples < 1.0:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1], run_details['average_length'][-1],
                                     run_details['average_fitness'][-1], run_details['best_length'][-1],
                                     run_details['best_fitness'][-1], oob_fitness, remaining_time))

    def fit(self, X, y):
        """Fit the Genetic Algo according to X, y.

        Parameters
        ----------
        X : a multi-index DataFrame, level 1 index is asset, level 2 index is dates of each asset,
            columns are raw common indicator.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to population_size (%d).'
                             % (self.hall_of_fame, self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to hall_of_fame (%d).'
                             % (self.n_components, self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if function not in _function_map:
                raise ValueError('invalid function name %s found in `function_set`.' % function)
            self._function_set.append(_function_map[function])
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover, self.p_subtree_mutation,
                                       self.p_hoist_mutation, self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.' % self.init_method)

        if not ((isinstance(self.const_range, tuple) and
                 len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical order: (min_depth, max_depth).')

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either "sigmoid" or _Function object, got %s'
                                 % type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, got %d.' % self._transformer.arity)
        if self.unit_dict is not None:
            self.unit_dict = unit_transform(self.unit_dict)
        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [], 'average_length': [], 'average_fitness': [], 'best_length': [],
                                 'best_fitness': [], 'best_oob_fitness': [], 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):

            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            n_jobs, n_programs, starts = _partition_estimators(self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)
            population = Parallel(n_jobs=n_jobs, verbose=int(self.verbose > 1))(delayed(
                _evolve)(n_programs[i], parents, X, y, seeds[starts[i]:starts[i + 1]], params) for i in range(n_jobs))
            population = list(itertools.chain.from_iterable(population))
            # Reduce, maintaining order across different n_jobs
            fitness = [program.fitness_ for program in population]
            length = [program.length_ for program in population]

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]
            best_fitness = best_program.fitness_
            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_fitness)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            if self._metric.greater_is_better:
                if best_fitness >= self.stopping_criteria or best_fitness <= 1.05 * self.pre_best:
                    break
            else:
                if best_fitness <= self.stopping_criteria or best_fitness >= 0.95 * self.pre_best:
                    break
            self.pre_best = best_fitness

        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute(X) for gp in [self._programs[-1][i] for i in hall_of_fame]])
            if self.metric == 'spearman':
                evaluation = np.apply_along_axis(rankdata, 1, evaluation)

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = np.abs(np.corrcoef(evaluation))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove the least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i] for i in hall_of_fame[components]]

        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

        return self


class SymbolicRegressor(BaseSymbolic, RegressorMixin):
    """A Genetic Programming symbolic regressor.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no constants will be included.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and terminals, allowing for smaller trees.
           Tends to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and then terminals are selected.
           Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and 'grow'.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs.

    metric : str, optional (default='mean absolute error')
        The name of the fitness metric.

    parsimony_coefficient : float, optional (default=0.001)
        This constant penalizes overfitting programs by adjusting their fitness to be less favorable for selection.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.

        Note : The above genetic operation probabilities must sum to less than 1.
        The balance of probability is assigned to 'reproduction', where a tournament winner is cloned.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in the `print` operation or `export_graphviz`.
        If None, then X0, X1, etc. will be used for representations.

    unit_dict : dict
        Dictionary of raw indicator's unit, used in program.unit_rationality.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)

    """

    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set,
                 metric,
                 parsimony_coefficient,
                 unit_dict,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            unit_dict=unit_dict,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(self, X):
        """Predict classes on test vectors X.

        Returns
        -------
        y : The predicted classes of the input samples.

        """
        return self._program.execute(X)


class SymbolicClassifier(BaseSymbolic, ClassifierMixin):
    """A Genetic Programming symbolic classifier.

    Parameters
    ----------
    transformer : str, optional (default='sigmoid')
        The name of the decision function.

    class_weight : dict or None, optional (default=None)
        class Weights in the form ``{class_label: weight}``. If not given, all classes are supposed to have weight 1.
    """

    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set,
                 transformer='sigmoid',
                 metric='log loss',
                 parsimony_coefficient,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicClassifier, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            transformer=transformer,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            class_weight=class_weight,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def _more_tags(self):
        return {'binary_only': True}

    def predict_proba(self, X):
        """Predict probabilities on test vectors X.

        Returns
        -------
        proba : array, shape = [n_samples, n_classes]
            The class probabilities of the input samples.
        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicClassifier not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        scores = self._program.execute(X)
        proba = self._transformer(scores)
        proba = np.vstack([1 - proba, proba]).T
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)


class SymbolicTransformer(BaseSymbolic, TransformerMixin):
    """A Genetic Programming symbolic transformer.

    Parameters
    ----------
    hall_of_fame : integer, or None, optional (default=100)
        The number of the fittest programs to compare when finding the least-correlated n_components individuals.
        If `None`, the entire final generation will be used.

    n_components : integer, or None, optional (default=10)
        The number of least-correlated programs to return after searching in the hall_of_fame.
        If `None`, the entire hall_of_fame will be used.
    """

    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=100,
                 n_components=10,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=1.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set,
                 metric='pearson',
                 parsimony_coefficient,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicTransformer, self).__init__(
            population_size=population_size,
            hall_of_fame=hall_of_fame,
            n_components=n_components,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __len__(self):
        """Overloads `len` output to be the number of fitted components."""
        if not hasattr(self, '_best_programs'):
            return 0
        return self.n_components

    def __getitem__(self, item):
        """Return the ith item of the fitted components."""
        if item >= len(self):
            raise IndexError
        return self._best_programs[item]

    def __str__(self):
        """Overloads `print` output of the object to resemble LISP trees."""
        if not hasattr(self, '_best_programs'):
            return self.__repr__()
        output = str([gp.__str__() for gp in self])
        return output.replace("',", ",\n").replace("'", "")

    def transform(self, X):
        """Transform X according to the fitted transformer.
        """
        if not hasattr(self, '_best_programs'):
            raise NotFittedError('SymbolicTransformer not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        X_new = np.array([gp.execute(X) for gp in self._best_programs]).T

        return X_new

    def fit_transform(self, X, y):
        """Fit to supply_demand_data, then transform it.
        """
        return self.fit(X, y).transform(X)
