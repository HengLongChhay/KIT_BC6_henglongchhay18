# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class _normalized_decision_function_radial_basis(__pybind11_builtins.pybind11_object):
    # no doc
    def batch_predict(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        batch_predict(*args, **kwargs)
        Overloaded function.
        
        1. batch_predict(self: dlib._normalized_decision_function_radial_basis, arg0: dlib.vectors) -> dlib.array
        
        2. batch_predict(self: dlib._normalized_decision_function_radial_basis, arg0: numpy.ndarray[(rows,cols),float64]) -> numpy.ndarray[float64]
        """
        pass

    def __call__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __call__(*args, **kwargs)
        Overloaded function.
        
        1. __call__(self: dlib._normalized_decision_function_radial_basis, arg0: dlib.vector) -> float
        
        2. __call__(self: dlib._normalized_decision_function_radial_basis, arg0: numpy.ndarray[float64]) -> float
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib._normalized_decision_function_radial_basis) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib._normalized_decision_function_radial_basis, arg0: tuple) -> None """
        pass

    alpha = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    b = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    basis_vectors = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    invstd_devs = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Input vectors are normalized by the equation, (x-means)*invstd_devs, before being passed to the underlying RBF function."""

    kernel_function = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    means = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Input vectors are normalized by the equation, (x-means)*invstd_devs, before being passed to the underlying RBF function."""



