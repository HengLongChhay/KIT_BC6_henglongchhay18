# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class simple_object_detector(__pybind11_builtins.pybind11_object):
    """ This object represents a sliding window histogram-of-oriented-gradients based object detector. """
    def run_multiple(self, detectors, image, upsample_num_times=0, adjust_threshold=0.0): # real signature unknown; restored from __doc__
        """
        run_multiple(detectors: list, image: array, upsample_num_times: int=0, adjust_threshold: float=0.0) -> tuple
        
        requires 
            - detectors is a list of detectors. 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - upsample_num_times >= 0 
        ensures 
            - This function runs the list of object detectors at once on the input image and returns 
              a tuple of (list of detections, list of scores, list of weight_indices).   
            - Upsamples the image upsample_num_times before running the basic 
              detector.
        """
        return ()

    def save(self, detector_output_filename): # real signature unknown; restored from __doc__
        """
        save(self: dlib.simple_object_detector, detector_output_filename: str) -> None
        
        Save a simple_object_detector to the provided path.
        """
        pass

    def __call__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __call__(*args, **kwargs)
        Overloaded function.
        
        1. __call__(self: dlib.simple_object_detector, image: array, upsample_num_times: int) -> dlib.rectangles
        
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - upsample_num_times >= 0 
        ensures 
            - This function runs the object detector on the input image and returns 
              a list of detections.   
            - Upsamples the image upsample_num_times before running the basic 
              detector.  If you don't know how many times you want to upsample then 
              don't provide a value for upsample_num_times and an appropriate 
              default will be used.
        
        2. __call__(self: dlib.simple_object_detector, image: array) -> dlib.rectangles
        
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
        ensures 
            - This function runs the object detector on the input image and returns 
              a list of detections.
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.simple_object_detector) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.simple_object_detector, detectors: list) -> None
        
        This version of the constructor builds a simple_object_detector from a 
        bunch of other simple_object_detectors.  It essentially packs them together 
        so that when you run the detector it's like calling run_multiple().  Except 
        in this case the non-max suppression is applied to them all as a group.  So 
        unlike run_multiple(), each detector competes in the non-max suppression. 
         
        Also, the non-max suppression settings used for this whole thing are 
        the settings used by detectors[0].  So if you have a preference,  
        put the detector that uses the type of non-max suppression you like first 
        in the list.
        
        2. __init__(self: dlib.simple_object_detector, arg0: str) -> None
        
        Loads a simple_object_detector from a file that contains the output of the 
        train_simple_object_detector() routine.
        """
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.simple_object_detector, arg0: tuple) -> None """
        pass

    detection_window_height = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    detection_window_width = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    num_detectors = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    upsampling_amount = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """The detector upsamples the image this many times before running."""



