simple_spearmint
----------------

A thin wrapper class around `spearmint <https://github.com/HIPS/Spearmint>`_,
which allows you to use it without setting up MongoDB, writing a config file,
creating a separate experiment script, etc. etc.

Example
-------

.. code-block:: python

  import simple_spearmint
  import numpy as np

  # Define a parameter space
  # Supported parameter types are 'int', 'float', and 'enum'
  parameter_space = {'x': {'type': 'float', 'min': -2, 'max': 2},
                     'y': {'type': 'int', 'min': 0, 'max': 3},
                     'function': {'type': 'enum', 'options': ['sin', 'cos']}}
  # Create an optimizer
  ss = simple_spearmint.SimpleSpearmint(parameter_space)

  # Define an objective function, must return a scalar value
  def objective(x, y, function):
      if function == 'sin':
          return np.sin(x)**y - y
      elif function == 'cos':
          return np.cos(x)**y - y

  # Seed with 5 randomly chosen parameter settings
  # (this step is optional, but can be beneficial)
  for n in xrange(5):
      # Get random parameter settings
      suggestion = ss.suggest_random()
      # Retrieve an objective value for these parameters
      value = objective(suggestion['x'],
                        suggestion['y'],
                        suggestion['function'])
      print "Random trial {}: {} -> {}".format(n + 1, suggestion, value)
      # Update the optimizer on the result
      ss.update(suggestion, value)

  # Run for 100 hyperparameter optimization trials
  for n in xrange(100):
      # Get a suggestion from the optimizer
      suggestion = ss.suggest()
      # Get an objective value; the ** syntax is equivalent to
      # the call to objective above
      value = objective(**suggestion)
      print "GP trial {}: {} -> {}".format(n + 1, suggestion, value)
      # Update the optimizer on the result
      ss.update(suggestion, value)
  best_parameters, best_objective = ss.get_best_parameters()
  print "Best parameters {} for objective {}".format(
      best_parameters, best_objective)
