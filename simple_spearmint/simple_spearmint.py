import spearmint.tasks.task_group
import spearmint.choosers.default_chooser
import os
import sys
import numpy as np


class SimpleSpearmint(object):
    """ Thin wrapper around Spearmint's Gaussian Process optimizer.

    Parameters
    ----------
    parameter_space : dict
        Dictionary defining the parameters to optimize over.  The keys should
        be parameter names and the values should be dictionaries which specify
        the parameter; for example, a parameter called "x" which is a float
        between -1 and 1 would be specified as
        ``'x': {'type': 'float', 'min': -1, 'max': 1}``.  Possible parameter
        types are ``'float'``, ``'int'``, and ``'enum'``.  For ``'float'`` and
        ``'int'``, ``'min'`` and ``'max'`` values must be supplied; for
        ``'enum'``, the possible parameter values must be supplied as a list
        with the key ``'options'``.

    noiseless : bool
        Whether the objective function is noiseless or not.  If the objective
        is noiseless, set ``noiseless=True``.

    debug : bool
        Whether to allow Spearmint to print debug information to stderr.

    Examples
    --------
    Create a parameter optimizer over three parameters: x, a float between -2
    and 2; y, an int between 0 and 3, and function, which can be either
    ``'sin'`` or ``'cos'``.

    >>> ss = simple_spearmint.SimpleSpearmint(
    ...     {'x': {'type': 'float', 'min': -2, 'max': 2},
    ...      'y': {'type': 'int', 'min': 0, 'max': 3},
    ...      'function': {'type': 'enum', 'options': ['sin', 'cos']}})
    ...
    >>> suggested_parameters = ss.suggest()

    """

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
        """ Converts parameter values in the form ``{'parameter_name': value}``
        to a spearmint-friendly format, which includes the key ``'type'`` and
        where ``'enum'`` variables have a list value.

        Parameters
        ----------
        parameter_values : dict
            Dictionary of the form ``{'parameter_name': value}``.

        Returns
        -------
        specd_parameter_values : dict
            Converted dictionary in the format expected by Spearmint's
            ``vectorify`` function.

        """
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
        """ Update the optimizer with a new result.

        Parameters
        ----------
        parameter_values : dict
            Dictionary mapping each parameter name to its value.

        objective_value : float
            The value of the objective function achieved by using these
            parameters.

        """
        self.parameter_values.append(parameter_values)
        self.objective_values.append(objective_value)
        self.task_group.inputs = np.array(
            [self.task_group.vectorify(self.spec_parameter_values(values))
             for values in self.parameter_values])
        self.task_group.values = {'main': np.array(self.objective_values)}

    def suggest(self):
        """ Generate a new parameter suggestion.

        Returns
        -------
        suggestion : dict
            Dictionary mapping parameter names to the suggested values.
        """
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
        """ Retrieve the best parameter values and objective for all trials.

        Returns
        -------
        best_parameters : dict
            Dictionary mapping parameter names to the suggested values
            corresponding to the trial with the lowest objective value.

        objective_value : float
            The lowest objective function value achieved.
        """
        best_objective = np.argmin(self.objective_values)
        return (self.parameter_values[best_objective],
                self.objective_values[best_objective])
