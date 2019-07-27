# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class shape_predictor(__pybind11_builtins.pybind11_object):
    """ This object is a tool that takes in an image region containing some object and outputs a set of point locations that define the pose of the object. The classic example of this is human face pose prediction, where you take an image of a human face as input and are expected to identify the locations of important facial landmarks such as the corners of the mouth and eyes, tip of the nose, and so forth. """
    def save(self, predictor_output_filename): # real signature unknown; restored from __doc__
        """
        save(self: dlib.shape_predictor, predictor_output_filename: str) -> None
        
        Save a shape_predictor to the provided path.
        """
        pass

    def __call__(self, image, box): # real signature unknown; restored from __doc__
        """
        __call__(self: dlib.shape_predictor, image: array, box: dlib.rectangle) -> dlib.full_object_detection
        
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - box is the bounding box to begin the shape prediction inside. 
        ensures 
            - This function runs the shape predictor on the input image and returns 
              a single full_object_detection.
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.shape_predictor) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.shape_predictor) -> None
        
        2. __init__(self: dlib.shape_predictor, arg0: str) -> None
        
        Loads a shape_predictor from a file that contains the output of the 
        train_shape_predictor() routine.
        """
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.shape_predictor, arg0: tuple) -> None """
        pass


