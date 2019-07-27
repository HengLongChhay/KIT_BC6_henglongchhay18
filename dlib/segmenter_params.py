# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class segmenter_params(__pybind11_builtins.pybind11_object):
    """
    This class is used to define all the optional parameters to the    
    train_sequence_segmenter() and cross_validate_sequence_segmenter() routines.
    """
    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.segmenter_params) -> tuple """
        return ()

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: dlib.segmenter_params) -> None """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.segmenter_params) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.segmenter_params, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.segmenter_params) -> str """
        return ""

    allow_negative_weights = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    be_verbose = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    C = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """SVM C parameter"""

    epsilon = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    max_cache_size = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    num_threads = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    use_BIO_model = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    use_high_order_features = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    window_size = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



