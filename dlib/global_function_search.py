# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class global_function_search(__pybind11_builtins.pybind11_object):
    """ See: http://dlib.net/dlib/global_optimization/global_function_search_abstract.h.html """
    def get_best_function_eval(self): # real signature unknown; restored from __doc__
        """ get_best_function_eval(self: dlib.global_function_search) -> tuple """
        return ()

    def get_function_evaluations(self): # real signature unknown; restored from __doc__
        """ get_function_evaluations(self: dlib.global_function_search) -> tuple """
        return ()

    def get_monte_carlo_upper_bound_sample_num(self): # real signature unknown; restored from __doc__
        """ get_monte_carlo_upper_bound_sample_num(self: dlib.global_function_search) -> int """
        return 0

    def get_next_x(self): # real signature unknown; restored from __doc__
        """ get_next_x(self: dlib.global_function_search) -> dlib.function_evaluation_request """
        pass

    def get_pure_random_search_probability(self): # real signature unknown; restored from __doc__
        """ get_pure_random_search_probability(self: dlib.global_function_search) -> float """
        return 0.0

    def get_relative_noise_magnitude(self): # real signature unknown; restored from __doc__
        """ get_relative_noise_magnitude(self: dlib.global_function_search) -> float """
        return 0.0

    def get_solver_epsilon(self): # real signature unknown; restored from __doc__
        """ get_solver_epsilon(self: dlib.global_function_search) -> float """
        return 0.0

    def num_functions(self): # real signature unknown; restored from __doc__
        """ num_functions(self: dlib.global_function_search) -> int """
        return 0

    def set_monte_carlo_upper_bound_sample_num(self, num): # real signature unknown; restored from __doc__
        """ set_monte_carlo_upper_bound_sample_num(self: dlib.global_function_search, num: int) -> None """
        pass

    def set_pure_random_search_probability(self, prob): # real signature unknown; restored from __doc__
        """ set_pure_random_search_probability(self: dlib.global_function_search, prob: float) -> None """
        pass

    def set_relative_noise_magnitude(self, value): # real signature unknown; restored from __doc__
        """ set_relative_noise_magnitude(self: dlib.global_function_search, value: float) -> None """
        pass

    def set_seed(self, seed): # real signature unknown; restored from __doc__
        """ set_seed(self: dlib.global_function_search, seed: int) -> None """
        pass

    def set_solver_epsilon(self, eps): # real signature unknown; restored from __doc__
        """ set_solver_epsilon(self: dlib.global_function_search, eps: float) -> None """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.global_function_search, function: dlib.function_spec) -> None
        
        2. __init__(self: dlib.global_function_search, functions: list) -> None
        
        3. __init__(self: dlib.global_function_search, functions: list, initial_function_evals: list, relative_noise_magnitude: float) -> None
        """
        pass


