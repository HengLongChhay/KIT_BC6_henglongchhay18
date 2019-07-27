# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class full_object_detection(__pybind11_builtins.pybind11_object):
    """ This object represents the location of an object in an image along with the     positions of each of its constituent parts. """
    def part(self, idx): # real signature unknown; restored from __doc__
        """
        part(self: dlib.full_object_detection, idx: int) -> dlib.point
        
        A single part of the object as a dlib point.
        """
        pass

    def parts(self): # real signature unknown; restored from __doc__
        """
        parts(self: dlib.full_object_detection) -> dlib.points
        
        A vector of dlib points representing all of the parts.
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.full_object_detection) -> tuple """
        return ()

    def __init__(self, arg0, arg1): # real signature unknown; restored from __doc__
        """
        __init__(self: dlib.full_object_detection, arg0: dlib.rectangle, arg1: list) -> None
        
        requires 
            - rect: dlib rectangle 
            - parts: list of dlib points
        """
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.full_object_detection, arg0: tuple) -> None """
        pass

    num_parts = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The number of parts of the object."""

    rect = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Bounding box from the underlying detector. Parts can be outside box if appropriate."""



