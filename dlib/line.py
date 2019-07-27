# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class line(__pybind11_builtins.pybind11_object):
    """
    This object represents a line in the 2D plane.  The line is defined by two points 
    running through it, p1 and p2.  This object also includes a unit normal vector that 
    is perpendicular to the line.
    """
    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.line) -> None
        
        p1, p2, and normal are all the 0 vector.
        
        2. __init__(self: dlib.line, a: dlib.dpoint, b: dlib.dpoint) -> None
        
        ensures 
            - #p1 == a 
            - #p2 == b 
            - #normal == A vector normal to the line passing through points a and b. 
              Therefore, the normal vector is the vector (a-b) but unit normalized and rotated clockwise 90 degrees.
        
        3. __init__(self: dlib.line, a: dlib.point, b: dlib.point) -> None
        
        ensures 
            - #p1 == a 
            - #p2 == b 
            - #normal == A vector normal to the line passing through points a and b. 
              Therefore, the normal vector is the vector (a-b) but unit normalized and rotated clockwise 90 degrees.
        """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.line) -> str """
        return ""

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.line) -> str """
        return ""

    normal = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """returns a unit vector that is normal to the line passing through p1 and p2."""

    p1 = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """returns the first endpoint of the line."""

    p2 = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """returns the second endpoint of the line."""



