# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class dpoint(__pybind11_builtins.pybind11_object):
    """ This object represents a single point of floating point coordinates that maps directly to a dlib::dpoint. """
    def normalize(self): # real signature unknown; restored from __doc__
        """
        normalize(self: dlib.dpoint) -> dlib.dpoint
        
        Returns a unit normalized copy of this vector.
        """
        pass

    def __add__(self, arg0): # real signature unknown; restored from __doc__
        """ __add__(self: dlib.dpoint, arg0: dlib.dpoint) -> dlib.dpoint """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.dpoint) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.dpoint, x: float, y: float) -> None
        
        2. __init__(self: dlib.dpoint, p: dlib.point) -> None
        
        3. __init__(self: dlib.dpoint, v: numpy.ndarray[int64]) -> None
        
        4. __init__(self: dlib.dpoint, v: numpy.ndarray[float32]) -> None
        
        5. __init__(self: dlib.dpoint, v: numpy.ndarray[float64]) -> None
        """
        pass

    def __mul__(self, arg0): # real signature unknown; restored from __doc__
        """ __mul__(self: dlib.dpoint, arg0: float) -> dlib.dpoint """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.dpoint) -> str """
        return ""

    def __rmul__(self, arg0): # real signature unknown; restored from __doc__
        """ __rmul__(self: dlib.dpoint, arg0: float) -> dlib.dpoint """
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.dpoint, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.dpoint) -> str """
        return ""

    def __sub__(self, arg0): # real signature unknown; restored from __doc__
        """ __sub__(self: dlib.dpoint, arg0: dlib.dpoint) -> dlib.dpoint """
        pass

    def __truediv__(self, arg0): # real signature unknown; restored from __doc__
        """ __truediv__(self: dlib.dpoint, arg0: float) -> dlib.dpoint """
        pass

    x = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The x-coordinate of the dpoint."""

    y = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The y-coordinate of the dpoint."""



