# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class shape_predictor_training_options(__pybind11_builtins.pybind11_object):
    """ This object is a container for the options to the train_shape_predictor() routine. """
    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.shape_predictor_training_options) -> tuple """
        return ()

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: dlib.shape_predictor_training_options) -> None """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.shape_predictor_training_options) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.shape_predictor_training_options, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.shape_predictor_training_options) -> str """
        return ""

    be_verbose = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """If true, train_shape_predictor() will print out a lot of information to stdout while training."""

    cascade_depth = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The number of cascades created to train the model with."""

    feature_pool_region_padding = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Size of region within which to sample features for the feature pool.                       positive values increase the sampling region while negative values decrease it. E.g. padding of 0 means we                       sample fr"""

    feature_pool_size = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Number of pixels used to generate features for the random trees."""

    lambda_param = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Controls how tight the feature sampling should be. Lower values enforce closer features."""

    landmark_relative_padding_mode = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """If True then features are drawn only from the box around the landmarks, otherwise they come from the bounding box and landmarks together.  See feature_pool_region_padding doc for more details."""

    nu = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The regularization parameter.  Larger values of this parameter                        will cause the algorithm to fit the training data better but may also                        cause overfitting.  The value must be in the range (0, 1]."""

    num_test_splits = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Number of split features at each node to sample. The one that gives the best split is chosen."""

    num_threads = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Use this many threads/CPU cores for training."""

    num_trees_per_cascade_level = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The number of trees created for each cascade."""

    oversampling_amount = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The number of randomly selected initial starting points sampled for each training example"""

    oversampling_translation_jitter = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The amount of translation jittering to apply to bounding boxes, a good value is in in the range [0 0.5]."""

    random_seed = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The random seed used by the internal random number generator"""

    tree_depth = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The depth of the trees used in each cascade. There are pow(2, get_tree_depth()) leaves in each tree"""



