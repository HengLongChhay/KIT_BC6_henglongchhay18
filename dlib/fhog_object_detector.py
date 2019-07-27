# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class fhog_object_detector(__pybind11_builtins.pybind11_object):
    """ This object represents a sliding window histogram-of-oriented-gradients based object detector. """
    def run(self, image, upsample_num_times=0, adjust_threshold=0.0): # real signature unknown; restored from __doc__
        """
        run(self: dlib.fhog_object_detector, image: array, upsample_num_times: int=0, adjust_threshold: float=0.0) -> tuple
        
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - upsample_num_times >= 0 
        ensures 
            - This function runs the object detector on the input image and returns 
              a tuple of (list of detections, list of scores, list of weight_indices).   
            - Upsamples the image upsample_num_times before running the basic 
              detector.
        """
        return ()

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
        save(self: dlib.fhog_object_detector, detector_output_filename: str) -> None
        
        Save a simple_object_detector to the provided path.
        """
        pass

    def __call__(self, image, upsample_num_times=0): # real signature unknown; restored from __doc__
        """
        __call__(self: dlib.fhog_object_detector, image: array, upsample_num_times: int=0) -> dlib.rectangles
        
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - upsample_num_times >= 0 
        ensures 
            - This function runs the object detector on the input image and returns 
              a list of detections.   
            - Upsamples the image upsample_num_times before running the basic 
              detector.
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.fhog_object_detector) -> tuple """
        return ()

    def __init__(self, arg0): # real signature unknown; restored from __doc__
        """
        __init__(self: dlib.fhog_object_detector, arg0: str) -> None
        
        Loads an object detector from a file that contains the output of the 
        train_simple_object_detector() routine or a serialized C++ object of type
        object_detector<scan_fhog_pyramid<pyramid_down<6>>>.
        """
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.fhog_object_detector, arg0: tuple) -> None """
        pass

    detection_window_height = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    detection_window_width = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    num_detectors = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



