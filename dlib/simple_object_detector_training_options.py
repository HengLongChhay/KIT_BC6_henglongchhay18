# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class simple_object_detector_training_options(__pybind11_builtins.pybind11_object):
    """ This object is a container for the options to the train_simple_object_detector() routine. """
    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: dlib.simple_object_detector_training_options) -> None """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.simple_object_detector_training_options) -> str """
        return ""

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.simple_object_detector_training_options) -> str """
        return ""

    add_left_right_image_flips = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """if true, train_simple_object_detector() will assume the objects are 
left/right symmetric and add in left right flips of the training 
images.  This doubles the size of the training dataset."""

    be_verbose = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """If true, train_simple_object_detector() will print out a lot of information to the screen while training."""

    C = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """C is the usual SVM C regularization parameter.  So it is passed to 
structural_object_detection_trainer::set_c().  Larger values of C 
will encourage the trainer to fit the data better but might lead to 
overfitting.  Therefore, you must determine the proper setting of 
this parameter experimentally."""

    detection_window_size = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The sliding window used will have about this many pixels inside it."""

    epsilon = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """epsilon is the stopping epsilon.  Smaller values make the trainer's 
solver more accurate but might take longer to train."""

    max_runtime_seconds = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Don't let the solver run for longer than this many seconds."""

    nuclear_norm_regularization_strength = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """This detector works by convolving a filter over a HOG feature image.  If that 
filter is separable then the convolution can be performed much faster.  The 
nuclear_norm_regularization_strength parameter encourages the machine learning 
algorithm to learn a separable filter.  A value of 0 disables this feature, but 
any non-zero value places a nuclear norm regularizer on the objective function 
and this encourages the learning of a separable filter.  Note that setting 
nuclear_norm_regularization_strength to a non-zero value can make the training 
process take significantly longer, so be patient when using it."""

    num_threads = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """train_simple_object_detector() will use this many threads of 
execution.  Set this to the number of CPU cores on your machine to 
obtain the fastest training speed."""

    upsample_limit = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """train_simple_object_detector() will upsample images if needed 
no more than upsample_limit times. Value 0 will forbid trainer to 
upsample any images. If trainer is unable to fit all boxes with 
required upsample_limit, exception will be thrown. Higher values 
of upsample_limit exponentially increases memory requirements. 
Values higher than 2 (default) are not recommended."""



