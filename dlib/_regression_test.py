# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class _regression_test(__pybind11_builtins.pybind11_object):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib._regression_test) -> str """
        return ""

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib._regression_test) -> str """
        return ""

    mean_average_error = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The mean average error of a regression function on a dataset."""

    mean_error_stddev = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The standard deviation of the absolute value of the error of a regression function on a dataset."""

    mean_squared_error = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The mean squared error of a regression function on a dataset."""

    R_squared = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """A value between 0 and 1, measures the squared correlation between the output of a 
regression function and the target values."""



