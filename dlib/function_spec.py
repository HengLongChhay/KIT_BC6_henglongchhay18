# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class function_spec(__pybind11_builtins.pybind11_object):
    """ See: http://dlib.net/dlib/global_optimization/global_function_search_abstract.h.html """
    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.function_spec, bound1: dlib.vector, bound2: dlib.vector) -> None
        
        2. __init__(self: dlib.function_spec, bound1: dlib.vector, bound2: dlib.vector, is_integer: List[bool]) -> None
        
        3. __init__(self: dlib.function_spec, bound1: list, bound2: list) -> None
        
        4. __init__(self: dlib.function_spec, bound1: list, bound2: list, is_integer: list) -> None
        """
        pass

    is_integer_variable = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    lower = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    upper = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



