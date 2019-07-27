# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class point_transform_projective(__pybind11_builtins.pybind11_object):
    """ This is an object that takes 2D points and applies a projective transformation to them. """
    def __call__(self, p): # real signature unknown; restored from __doc__
        """
        __call__(self: dlib.point_transform_projective, p: dlib.dpoint) -> dlib.dpoint
        
        ensures 
            - Applies the projective transformation defined by this object's constructor 
              to p and returns the result.  To define this precisely: 
                - let p_h == the point p in homogeneous coordinates.  That is: 
                    - p_h.x == p.x 
                    - p_h.y == p.y 
                    - p_h.z == 1  
                - let x == m*p_h  
                - Then this function returns the value x/x.z
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.point_transform_projective) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.point_transform_projective) -> None
        
        ensures 
            - This object will perform the identity transform.  That is, given a point 
              as input it will return the same point as output.  Therefore, self.m == a 3x3 identity matrix.
        
        2. __init__(self: dlib.point_transform_projective, m: numpy.ndarray[(rows,cols),float64]) -> None
        
        ensures 
            - self.m == m
        """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.point_transform_projective) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.point_transform_projective, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.point_transform_projective) -> str """
        return ""

    m = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """m is the 3x3 matrix that defines the projective transformation."""



