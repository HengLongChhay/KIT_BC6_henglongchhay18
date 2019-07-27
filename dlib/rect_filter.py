# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class rect_filter(__pybind11_builtins.pybind11_object):
    """
    This object is a simple tool for filtering a rectangle that
                    measures the location of a moving object that has some non-trivial
                    momentum.  Importantly, the measurements are noisy and the object can
                    experience sudden unpredictable accelerations.  To accomplish this
                    filtering we use a simple Kalman filter with a state transition model of:
    
                        position_{i+1} = position_{i} + velocity_{i} 
                        velocity_{i+1} = velocity_{i} + some_unpredictable_acceleration
    
                    and a measurement model of:
                        
                        measured_position_{i} = position_{i} + measurement_noise
    
                    Where some_unpredictable_acceleration and measurement_noise are 0 mean Gaussian 
                    noise sources with standard deviations of typical_acceleration and
                    measurement_noise respectively.
    
                    To allow for really sudden and large but infrequent accelerations, at each
                    step we check if the current measured position deviates from the predicted
                    filtered position by more than max_measurement_deviation*measurement_noise 
                    and if so we adjust the filter's state to keep it within these bounds.
                    This allows the moving object to undergo large unmodeled accelerations, far
                    in excess of what would be suggested by typical_acceleration, without
                    then experiencing a long lag time where the Kalman filter has to "catches
                    up" to the new position.
    """
    def max_measurement_deviation(self): # real signature unknown; restored from __doc__
        """ max_measurement_deviation(self: dlib.rect_filter) -> float """
        return 0.0

    def measurement_noise(self): # real signature unknown; restored from __doc__
        """ measurement_noise(self: dlib.rect_filter) -> float """
        return 0.0

    def typical_acceleration(self): # real signature unknown; restored from __doc__
        """ typical_acceleration(self: dlib.rect_filter) -> float """
        return 0.0

    def __call__(self, rect): # real signature unknown; restored from __doc__
        """ __call__(self: dlib.rect_filter, rect: dlib.rectangle) -> dlib.rectangle """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.rect_filter) -> tuple """
        return ()

    def __init__(self, measurement_noise, typical_acceleration, max_measurement_deviation): # real signature unknown; restored from __doc__
        """ __init__(self: dlib.rect_filter, measurement_noise: float, typical_acceleration: float, max_measurement_deviation: float) -> None """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.rect_filter) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.rect_filter, arg0: tuple) -> None """
        pass


