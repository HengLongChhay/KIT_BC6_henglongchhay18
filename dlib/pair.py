# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class pair(__pybind11_builtins.pybind11_object):
    """ This object is used to represent the elements of a sparse_vector. """
    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.pair) -> tuple """
        return ()

    def __init__(self, arg0, arg1): # real signature unknown; restored from __doc__
        """ __init__(self: dlib.pair, arg0: int, arg1: float) -> None """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.pair) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.pair, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.pair) -> str """
        return ""

    first = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """This field represents the index/dimension number."""

    second = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """This field contains the value in a vector at dimension specified by the first field."""



