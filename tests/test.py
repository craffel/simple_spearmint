from simple_spearmint import SimpleSpearmint
import numpy as np


# Define an objective function, must return a scalar value
def squared(x):
    return x ** 2

def negative_squared(x):
    return -1.0 * (x ** 2)

# test to see if the minimum of f(x) = x^2 is the same point that maximizes f(x) = -x^2
def test_maximize():
    
    min_ss = SimpleSpearmint({'x': {'type': 'float', 'min': -3, 'max': 3}})
    max_ss = SimpleSpearmint({'x': {'type': 'float', 'min': -3, 'max': 3}}, minimize=False)
    
    # minimize f(x) = x^2
    # Run for 100 hyperparameter optimization trials
    for n in range(100):
        print("Running iteration ", n)
        # Get a suggestion from the optimizer
        min_suggestion = min_ss.suggest()
        max_suggestion = max_ss.suggest()
        
        # Get an objective value; the ** syntax is equivalent to
        # the call to objective above
        min_ss_value = squared(**min_suggestion)
        max_ss_value = negative_squared(**max_suggestion)
        
        # Update the optimizer on the result
        min_ss.update(min_suggestion, min_ss_value)
        max_ss.update(max_suggestion, max_ss_value)
        
    
    
    best_min_parameters, best_min_objective = min_ss.get_best_parameters()
    best_max_parameters, best_max_objective = max_ss.get_best_parameters()
    print("Done.  Best min value was ", best_min_objective, " best max value was ", best_max_objective)
    
    # test to see if the best params are roughly equal
    assert(np.allclose(np.asarray([best_min_parameters['x']]), np.asarray([best_max_parameters['x']])))
    
if __name__ == '__main__':
    test_maximize()