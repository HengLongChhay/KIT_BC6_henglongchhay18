# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class function_evaluation(__pybind11_builtins.pybind11_object):
    """
    This object records the output of a real valued function in response to
    some input. 
    
    In particular, if you have a function F(x) then the function_evaluation is
    simply a struct that records x and the scalar value F(x).
    """
    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.function_evaluation, x: dlib.vector, y: float) -> None
        
        2. __init__(self: dlib.function_evaluation, x: list, y: float) -> None
        """
        pass

    x = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    y = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



