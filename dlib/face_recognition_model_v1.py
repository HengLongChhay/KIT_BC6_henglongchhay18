# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class face_recognition_model_v1(__pybind11_builtins.pybind11_object):
    """ This object maps human faces into 128D vectors where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart.  The constructor loads the face recognition model from a file. The model file is available here: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 """
    def compute_face_descriptor(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        compute_face_descriptor(*args, **kwargs)
        Overloaded function.
        
        1. compute_face_descriptor(self: dlib.face_recognition_model_v1, img: numpy.ndarray[(rows,cols,3),uint8], face: dlib.full_object_detection, num_jitters: int=0, padding: float=0.25) -> dlib.vector
        
        Takes an image and a full_object_detection that references a face in that image and converts it into a 128D face descriptor. If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.
        
        2. compute_face_descriptor(self: dlib.face_recognition_model_v1, img: numpy.ndarray[(rows,cols,3),uint8], num_jitters: int=0) -> dlib.vector
        
        Takes an aligned face image of size 150x150 and converts it into a 128D face descriptor.Note that the alignment should be done in the same way dlib.get_face_chip does it.If num_jitters>1 then image will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. 
        
        3. compute_face_descriptor(self: dlib.face_recognition_model_v1, img: numpy.ndarray[(rows,cols,3),uint8], faces: dlib.full_object_detections, num_jitters: int=0, padding: float=0.25) -> dlib.vectors
        
        Takes an image and an array of full_object_detections that reference faces in that image and converts them into 128D face descriptors.  If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.
        
        4. compute_face_descriptor(self: dlib.face_recognition_model_v1, batch_img: List[numpy.ndarray[(rows,cols,3),uint8]], batch_faces: List[dlib.full_object_detections], num_jitters: int=0, padding: float=0.25) -> dlib.vectorss
        
        Takes an array of images and an array of arrays of full_object_detections. `batch_faces[i]` must be an array of full_object_detections corresponding to the image `batch_img[i]`, referencing faces in that image. Every face will be converted into 128D face descriptors.  If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.
        
        5. compute_face_descriptor(self: dlib.face_recognition_model_v1, batch_img: List[numpy.ndarray[(rows,cols,3),uint8]], num_jitters: int=0) -> dlib.vectors
        
        Takes an array of aligned images of faces of size 150_x_150.Note that the alignment should be done in the same way dlib.get_face_chip does it.Every face will be converted into 128D face descriptors.  If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor.
        """
        pass

    def __init__(self, arg0): # real signature unknown; restored from __doc__
        """ __init__(self: dlib.face_recognition_model_v1, arg0: str) -> None """
        pass


