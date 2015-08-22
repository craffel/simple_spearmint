import spearmint.tasks.task_group
import spearmint.choosers.default_chooser
import os
import sys
import numpy as np


class SimpleSpearmint(object):
    def __init__(self, parameter_space, noiseless=False, debug=False):
        for name, spec in parameter_space.items():
            spec['size'] = 1
            parameter_space[name] = spec
        noiseless = 'NOISELESS' if noiseless else 'GAUSSIAN'
        self.task_config = {'main': {'type': 'objective',
                                     'likelihood': noiseless}}
        self.task_group = spearmint.tasks.task_group.TaskGroup(
            self.task_config, parameter_space)
        self.chooser = spearmint.choosers.default_chooser.init({})
        self.parameter_values = []
        self.objective_values = []
        self.hypers = None
        self.debug = debug

    def spec_parameter_values(self, parameter_values):
        specd_parameter_values = {}
        for name, value in parameter_values.items():
            param_type = self.task_group.variables_config[name]['type']
            if param_type == 'enum':
                values = [value]
            else:
                values = value
            specd_parameter_values[name] = {'type': param_type,
                                            'values': values}
        return specd_parameter_values

    def update(self, parameter_values, objective_value):
        self.parameter_values.append(parameter_values)
        self.objective_values.append(objective_value)
        self.task_group.inputs = np.array(
            [self.task_group.vectorify(self.spec_parameter_values(values))
             for values in self.parameter_values])
        self.task_group.values = {'main': np.array(self.objective_values)}

    def suggest(self):
        if not self.debug:
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
        self.hypers = self.chooser.fit(
            self.task_group, self.hypers, self.task_config)
        suggestion = self.chooser.suggest()
        if not self.debug:
            sys.stderr.close()
            sys.stderr = old_stderr
        suggestion = self.task_group.paramify(np.atleast_1d(suggestion))
        suggestion = dict((name, value['values'][0])
                          for name, value in suggestion.items())
        return suggestion

    def get_best_parameters(self):
        best_objective = np.argmin(self.objective_values)
        return (self.parameter_values[best_objective],
                self.objective_values[best_objective])
