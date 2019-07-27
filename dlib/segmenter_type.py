# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class segmenter_type(__pybind11_builtins.pybind11_object):
    """ This object represents a sequence segmenter and is the type of object returned by the dlib.train_sequence_segmenter() routine. """
    def __call__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __call__(*args, **kwargs)
        Overloaded function.
        
        1. __call__(self: dlib.segmenter_type, arg0: dlib.vectors) -> dlib.ranges
        
        2. __call__(self: dlib.segmenter_type, arg0: dlib.sparse_vectors) -> dlib.ranges
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.segmenter_type) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.segmenter_type, arg0: tuple) -> None """
        pass

    weights = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



