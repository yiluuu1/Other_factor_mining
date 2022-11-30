from copy import copy
import numpy as np
import pandas as pd
from sklearn.utils.random import sample_without_replacement
from .my_functions import _Function
from .my_utils import check_random_state, preprocess, check_unit


class _Program(object):
    """A program-like representation of the evolved program.
    This is the underlying supply_demand_data-structure used by the public classes.
    """

    def __init__(self, function_set, arities, init_depth, init_method, n_features, const_range, metric, p_point_replace,
                 parsimony_coefficient, random_state, unit_dict, transformer=None, feature_names=None, program=None):
        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.unit_dict = unit_dict
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create an initial random program
            self.program = self.build_program(random_state)

        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        """Build an initial random program.
        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            # Determine if we are adding a function or terminal
            if (len(terminal_stack) < max_depth) and (method == 'full' or random_state.randint(
                    self.n_features + len(self.function_set)) <= len(self.function_set)):
                function = self.function_set[random_state.randint(len(self.function_set))]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is None:
                    terminal = random_state.randint(self.n_features)
                else:
                    terminal = random_state.randint(self.n_features + 1)
                    if terminal == self.n_features:
                        terminal = random_state.uniform(*self.const_range)
                program.append(terminal)
                terminal_stack[-1] -= 1

                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

    def validate_program(self):
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
                if node.extra_param is not None:
                    output += str(node.extra_param) + ','
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def _depth(self):
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.
        """

        node = self.program[0]  # Check for single-node programs
        if isinstance(node, float):
            return preprocess(np.repeat(node, X.shape[0]))
        if isinstance(node, int):
            return preprocess(X.iloc[:, node])

        apply_stack = []

        for node in self.program:
            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [pd.Series(np.repeat(t, X.shape[0]), index=X.index) if isinstance(t, float)
                             else X.iloc[:, t] if isinstance(t, int) else t for t in apply_stack[-1][1:]]
                if function.extra_param is not None:
                    if isinstance(function.extra_param, tuple):
                        function.extra_param = np.random.randint(*function.extra_param)
                    terminals.append(function.extra_param)

                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return preprocess(intermediate_result)

    def get_all_indices(self, n_samples=None, max_samples=None, random_state=None):
        """Get the indices on which to evaluate the fitness of a program.
        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(self._n_samples, self._n_samples - self._max_samples,
                                                 random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def fitness(self, X, y):
        """Evaluate the penalized fitness of the program according to X, y.
        """
        # if self.transformer:
        #     y_pred = self.transformer(y_pred)
        return self.metric(y, self.execute(X)) - self.parsimony_coefficient * (self.length_ - 1) * self.metric.sign

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.
        """
        if program is None:
            program = self.program
        # Reference Koza's(1992): choosing functions 90% of prob.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1 for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.
        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) - set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] + donor[donor_start:donor_end] + self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.
        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.
        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) - set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.
        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) < self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        raise ValueError('A constant was produced with const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    def unit_rationality(self, X):
        """Perform unit rationality check for every single factorof program."""
        rationality = ['weight', 'money', 'time', 'area', 'volume', 'price', 'money/weight', 'weight/time',
                       'weight/area', 'money/volume', 'volume/time', 'volume/area', None]
        node = self.program[0]
        if isinstance(node, (int, float)):
            return True

        apply_stack = []

        for node in self.program:
            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0].name
                arg = apply_stack[-1][0].extra_param
                terminals = [None if isinstance(t, float) else self.unit_dict[
                    X.columns[t]] if isinstance(t, int) else t for t in apply_stack[-1][1:]]

                intermediate_result = check_unit(function, terminals, arg)
                if intermediate_result == 'wrong':
                    return False
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result in rationality

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
