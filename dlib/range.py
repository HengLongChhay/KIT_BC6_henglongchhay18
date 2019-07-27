# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class range(__pybind11_builtins.pybind11_object):
    """ This object is used to represent a range of elements in an array. """
    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.range) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.range, arg0: int, arg1: int) -> None
        
        2. __init__(self: dlib.range, arg0: int) -> None
        """
        pass

    def __iter__(self): # real signature unknown; restored from __doc__
        """ __iter__(self: dlib.range) -> range_iter """
        pass

    def __len__(self): # real signature unknown; restored from __doc__
        """ __len__(self: dlib.range) -> int """
        return 0

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.range) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.range, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.range) -> str """
        return ""

    begin = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The index of the first element in the range.  This is represented using an unsigned integer."""

    end = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """One past the index of the last element in the range.  This is represented using an unsigned integer."""



