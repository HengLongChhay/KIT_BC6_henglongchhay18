# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class correlation_tracker(__pybind11_builtins.pybind11_object):
    """
    This is a tool for tracking moving objects in a video stream.  You give it 
                the bounding box of an object in the first frame and it attempts to track the 
                object in the box from frame to frame.  
                This tool is an implementation of the method described in the following paper: 
                    Danelljan, Martin, et al. 'Accurate scale estimation for robust visual 
                    tracking.' Proceedings of the British Machine Vision Conference BMVC. 2014.
    """
    def get_position(self): # real signature unknown; restored from __doc__
        """
        get_position(self: dlib.correlation_tracker) -> dlib.drectangle
        
        returns the predicted position of the object under track.
        """
        pass

    def start_track(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        start_track(*args, **kwargs)
        Overloaded function.
        
        1. start_track(self: dlib.correlation_tracker, image: array, bounding_box: dlib.drectangle) -> None
        
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - bounding_box.is_empty() == false 
                    ensures 
                        - This object will start tracking the thing inside the bounding box in the 
                          given image.  That is, if you call update() with subsequent video frames 
                          then it will try to keep track of the position of the object inside bounding_box. 
                        - #get_position() == bounding_box
        
        2. start_track(self: dlib.correlation_tracker, image: array, bounding_box: dlib.rectangle) -> None
        
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - bounding_box.is_empty() == false 
                    ensures 
                        - This object will start tracking the thing inside the bounding box in the 
                          given image.  That is, if you call update() with subsequent video frames 
                          then it will try to keep track of the position of the object inside bounding_box. 
                        - #get_position() == bounding_box
        """
        pass

    def update(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        update(*args, **kwargs)
        Overloaded function.
        
        1. update(self: dlib.correlation_tracker, image: array) -> float
        
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - get_position().is_empty() == false 
                          (i.e. you must have started tracking by calling start_track()) 
                    ensures 
                        - performs: return update(img, get_position())
        
        2. update(self: dlib.correlation_tracker, image: array, guess: dlib.drectangle) -> float
        
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - get_position().is_empty() == false 
                          (i.e. you must have started tracking by calling start_track()) 
                    ensures 
                        - When searching for the object in img, we search in the area around the 
                          provided guess. 
                        - #get_position() == the new predicted location of the object in img.  This 
                          location will be a copy of guess that has been translated and scaled 
                          appropriately based on the content of img so that it, hopefully, bounds 
                          the object in img. 
                        - Returns the peak to side-lobe ratio.  This is a number that measures how 
                          confident the tracker is that the object is inside #get_position(). 
                          Larger values indicate higher confidence.
        
        3. update(self: dlib.correlation_tracker, image: array, guess: dlib.rectangle) -> float
        
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - get_position().is_empty() == false 
                          (i.e. you must have started tracking by calling start_track()) 
                    ensures 
                        - When searching for the object in img, we search in the area around the 
                          provided guess. 
                        - #get_position() == the new predicted location of the object in img.  This 
                          location will be a copy of guess that has been translated and scaled 
                          appropriately based on the content of img so that it, hopefully, bounds 
                          the object in img. 
                        - Returns the peak to side-lobe ratio.  This is a number that measures how 
                          confident the tracker is that the object is inside #get_position(). 
                          Larger values indicate higher confidence.
        """
        pass

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: dlib.correlation_tracker) -> None """
        pass


