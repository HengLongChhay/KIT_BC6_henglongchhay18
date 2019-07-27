# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


# Variables with simple values

DLIB_USE_BLAS = True
DLIB_USE_CUDA = False
DLIB_USE_LAPACK = True

USE_AVX_INSTRUCTIONS = True

USE_NEON_INSTRUCTIONS = False

__time_compiled__ = 'Jul 23 2019 15:31:02'

__version__ = '19.17.0'

# functions

def angle_between_lines(a, b): # real signature unknown; restored from __doc__
    """
    angle_between_lines(a: dlib.line, b: dlib.line) -> float
    
    ensures 
        - returns the angle, in degrees, between the given lines.  This is a number in 
          the range [0 90].
    """
    return 0.0

def apply_cca_transform(m, v): # real signature unknown; restored from __doc__
    """
    apply_cca_transform(m: dlib.matrix, v: dlib.sparse_vector) -> dlib.vector
    
    requires    
        - max_index_plus_one(v) <= m.nr()    
    ensures    
        - returns trans(m)*v    
          (i.e. multiply m by the vector v and return the result)
    """
    pass

def assignment_cost(cost, assignment): # real signature unknown; restored from __doc__
    """
    assignment_cost(cost: dlib.matrix, assignment: list) -> float
    
    requires    
        - cost.nr() == cost.nc()    
          (i.e. the input must be a square matrix)    
        - for all valid i:    
            - 0 <= assignment[i] < cost.nr()    
    ensures    
        - Interprets cost as a cost assignment matrix. That is, cost[i][j]     
          represents the cost of assigning i to j.      
        - Interprets assignment as a particular set of assignments. That is,    
          i is assigned to assignment[i].    
        - returns the cost of the given assignment. That is, returns    
          a number which is:    
            sum over i: cost[i][assignment[i]]
    """
    return 0.0

def as_grayscale(img): # real signature unknown; restored from __doc__
    """
    as_grayscale(img: array) -> array
    
    Convert an image to 8bit grayscale.  If it's already a grayscale image do nothing and just return img.
    """
    return array

def auto_train_rbf_classifier(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    auto_train_rbf_classifier(*args, **kwargs)
    Overloaded function.
    
    1. auto_train_rbf_classifier(x: dlib.vectors, y: dlib.array, max_runtime_seconds: float, be_verbose: bool=True) -> dlib._normalized_decision_function_radial_basis
    
    requires 
        - y contains at least 6 examples of each class.  Moreover, every element in y 
          is either +1 or -1. 
        - max_runtime_seconds >= 0 
        - len(x) == len(y) 
        - all the vectors in x have the same dimension. 
    ensures 
        - This routine trains a radial basis function SVM on the given binary 
          classification training data.  It uses the svm_c_trainer to do this.  It also 
          uses find_max_global() and 6-fold cross-validation to automatically determine 
          the best settings of the SVM's hyper parameters. 
        - Note that we interpret y[i] as the label for the vector x[i].  Therefore, the 
          returned function, df, should generally satisfy sign(df(x[i])) == y[i] as 
          often as possible. 
        - The hyperparameter search will run for about max_runtime and will print 
          messages to the screen as it runs if be_verbose==true.
    
    2. auto_train_rbf_classifier(x: numpy.ndarray[(rows,cols),float64], y: numpy.ndarray[float64], max_runtime_seconds: float, be_verbose: bool=True) -> dlib._normalized_decision_function_radial_basis
    
    requires 
        - y contains at least 6 examples of each class.  Moreover, every element in y 
          is either +1 or -1. 
        - max_runtime_seconds >= 0 
        - len(x.shape(0)) == len(y) 
        - x.shape(1) > 0 
    ensures 
        - This routine trains a radial basis function SVM on the given binary 
          classification training data.  It uses the svm_c_trainer to do this.  It also 
          uses find_max_global() and 6-fold cross-validation to automatically determine 
          the best settings of the SVM's hyper parameters. 
        - Note that we interpret y[i] as the label for the vector x[i].  Therefore, the 
          returned function, df, should generally satisfy sign(df(x[i])) == y[i] as 
          often as possible. 
        - The hyperparameter search will run for about max_runtime and will print 
          messages to the screen as it runs if be_verbose==true.
    """
    pass

def cca(L, R, num_correlations, extra_rank=5, q=2, regularization=0): # real signature unknown; restored from __doc__
    """
    cca(L: dlib.sparse_vectors, R: dlib.sparse_vectors, num_correlations: int, extra_rank: int=5, q: int=2, regularization: float=0) -> dlib.cca_outputs
    
    requires    
        - num_correlations > 0    
        - len(L) > 0     
        - len(R) > 0     
        - len(L) == len(R)    
        - regularization >= 0    
        - L and R must be properly sorted sparse vectors.  This means they must list their  
          elements in ascending index order and not contain duplicate index values.  You can use 
          make_sparse_vector() to ensure this is true.  
    ensures    
        - This function performs a canonical correlation analysis between the vectors    
          in L and R.  That is, it finds two transformation matrices, Ltrans and    
          Rtrans, such that row vectors in the transformed matrices L*Ltrans and    
          R*Rtrans are as correlated as possible (note that in this notation we    
          interpret L as a matrix with the input vectors in its rows).  Note also that    
          this function tries to find transformations which produce num_correlations    
          dimensional output vectors.    
        - Note that you can easily apply the transformation to a vector using     
          apply_cca_transform().  So for example, like this:     
            - apply_cca_transform(Ltrans, some_sparse_vector)    
        - returns a structure containing the Ltrans and Rtrans transformation matrices    
          as well as the estimated correlations between elements of the transformed    
          vectors.    
        - This function assumes the data vectors in L and R have already been centered    
          (i.e. we assume the vectors have zero means).  However, in many cases it is    
          fine to use uncentered data with cca().  But if it is important for your    
          problem then you should center your data before passing it to cca().   
        - This function works with reduced rank approximations of the L and R matrices.    
          This makes it fast when working with large matrices.  In particular, we use    
          the dlib::svd_fast() routine to find reduced rank representations of the input    
          matrices by calling it as follows: svd_fast(L, U,D,V, num_correlations+extra_rank, q)     
          and similarly for R.  This means that you can use the extra_rank and q    
          arguments to cca() to influence the accuracy of the reduced rank    
          approximation.  However, the default values should work fine for most    
          problems.    
        - The dimensions of the output vectors produced by L*#Ltrans or R*#Rtrans are 
          ordered such that the dimensions with the highest correlations come first. 
          That is, after applying the transforms produced by cca() to a set of vectors 
          you will find that dimension 0 has the highest correlation, then dimension 1 
          has the next highest, and so on.  This also means that the list of estimated 
          correlations returned from cca() will always be listed in decreasing order. 
        - This function performs the ridge regression version of Canonical Correlation    
          Analysis when regularization is set to a value > 0.  In particular, larger    
          values indicate the solution should be more heavily regularized.  This can be    
          useful when the dimensionality of the data is larger than the number of    
          samples.    
        - A good discussion of CCA can be found in the paper "Canonical Correlation    
          Analysis" by David Weenink.  In particular, this function is implemented    
          using equations 29 and 30 from his paper.  We also use the idea of doing CCA    
          on a reduced rank approximation of L and R as suggested by Paramveer S.    
          Dhillon in his paper "Two Step CCA: A new spectral method for estimating    
          vector models of words".
    """
    pass

def center(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    center(*args, **kwargs)
    Overloaded function.
    
    1. center(rect: dlib.rectangle) -> dlib.point
    
        returns the center of the given rectangle
    
    2. center(rect: dlib.drectangle) -> dlib.dpoint
    
        returns the center of the given rectangle
    """
    pass

def centered_rect(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    centered_rect(*args, **kwargs)
    Overloaded function.
    
    1. centered_rect(p: dlib.point, width: int, height: int) -> dlib.rectangle
    
    2. centered_rect(p: dlib.dpoint, width: int, height: int) -> dlib.rectangle
    
    3. centered_rect(rect: dlib.rectangle, width: int, height: int) -> dlib.rectangle
    
    4. centered_rect(rect: dlib.drectangle, width: int, height: int) -> dlib.rectangle
    """
    pass

def centered_rects(pts, width, height): # real signature unknown; restored from __doc__
    """ centered_rects(pts: dlib.points, width: int, height: int) -> dlib.rectangles """
    pass

def chinese_whispers(edges): # real signature unknown; restored from __doc__
    """
    chinese_whispers(edges: list) -> list
    
    Given a graph with vertices represented as numbers indexed from 0, this algorithm takes a list of edges and returns back a list that contains a labels (found clusters) for each vertex. Edges are tuples with either 2 elements (integers presenting indexes of connected vertices) or 3 elements, where additional one element is float which presents distance weight of the edge). Offers direct access to dlib::chinese_whispers.
    """
    return []

def chinese_whispers_clustering(descriptors, threshold): # real signature unknown; restored from __doc__
    """
    chinese_whispers_clustering(descriptors: list, threshold: float) -> list
    
    Takes a list of descriptors and returns a list that contains a label for each descriptor. Clustering is done using dlib::chinese_whispers.
    """
    return []

def convert_image(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    convert_image(*args, **kwargs)
    Overloaded function.
    
    1. convert_image(img: numpy.ndarray[(rows,cols),uint8], dtype: str) -> array
    
    2. convert_image(img: numpy.ndarray[(rows,cols),uint16], dtype: str) -> array
    
    3. convert_image(img: numpy.ndarray[(rows,cols),uint32], dtype: str) -> array
    
    4. convert_image(img: numpy.ndarray[(rows,cols),uint64], dtype: str) -> array
    
    5. convert_image(img: numpy.ndarray[(rows,cols),int8], dtype: str) -> array
    
    6. convert_image(img: numpy.ndarray[(rows,cols),int16], dtype: str) -> array
    
    7. convert_image(img: numpy.ndarray[(rows,cols),int32], dtype: str) -> array
    
    8. convert_image(img: numpy.ndarray[(rows,cols),int64], dtype: str) -> array
    
    9. convert_image(img: numpy.ndarray[(rows,cols),float32], dtype: str) -> array
    
    10. convert_image(img: numpy.ndarray[(rows,cols),float64], dtype: str) -> array
    
    11. convert_image(img: numpy.ndarray[(rows,cols,3),uint8], dtype: str) -> array
    
    Converts an image to a target pixel type.  dtype must be a string containing one of the following: 
        uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel 
     
    When converting from a color space with more than 255 values the pixel intensity is 
    saturated at the minimum and maximum pixel values of the target pixel type.  For 
    example, if you convert a float valued image to uint8 then float values will be 
    truncated to integers and values larger than 255 are converted to 255 while values less 
    than 0 are converted to 0.
    """
    pass

def convert_image_scaled(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    convert_image_scaled(*args, **kwargs)
    Overloaded function.
    
    1. convert_image_scaled(img: numpy.ndarray[(rows,cols),uint8], dtype: str, thresh: float=4) -> array
    
    2. convert_image_scaled(img: numpy.ndarray[(rows,cols),uint16], dtype: str, thresh: float=4) -> array
    
    3. convert_image_scaled(img: numpy.ndarray[(rows,cols),uint32], dtype: str, thresh: float=4) -> array
    
    4. convert_image_scaled(img: numpy.ndarray[(rows,cols),uint64], dtype: str, thresh: float=4) -> array
    
    5. convert_image_scaled(img: numpy.ndarray[(rows,cols),int8], dtype: str, thresh: float=4) -> array
    
    6. convert_image_scaled(img: numpy.ndarray[(rows,cols),int16], dtype: str, thresh: float=4) -> array
    
    7. convert_image_scaled(img: numpy.ndarray[(rows,cols),int32], dtype: str, thresh: float=4) -> array
    
    8. convert_image_scaled(img: numpy.ndarray[(rows,cols),int64], dtype: str, thresh: float=4) -> array
    
    9. convert_image_scaled(img: numpy.ndarray[(rows,cols),float32], dtype: str, thresh: float=4) -> array
    
    10. convert_image_scaled(img: numpy.ndarray[(rows,cols),float64], dtype: str, thresh: float=4) -> array
    
    11. convert_image_scaled(img: numpy.ndarray[(rows,cols,3),uint8], dtype: str, thresh: float=4) -> array
    
    requires 
        - thresh > 0 
    ensures 
        - Converts an image to a target pixel type.  dtype must be a string containing one of the following: 
          uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel 
     
          The contents of img will be scaled to fit the dynamic range of the target 
          pixel type.  The thresh parameter is used to filter source pixel values which 
          are outliers.  These outliers will saturate at the edge of the destination 
          image's dynamic range. 
        - Specifically, for all valid r and c: 
            - We scale img[r][c] into the dynamic range of the target pixel type.  This 
              is done using the mean and standard deviation of img. Call the mean M and 
              the standard deviation D.  Then the scaling from source to destination is 
              performed using the following mapping: 
                let SRC_UPPER  = min(M + thresh*D, max(img)) 
                let SRC_LOWER  = max(M - thresh*D, min(img)) 
                let DEST_UPPER = max value possible for the selected dtype.  
                let DEST_LOWER = min value possible for the selected dtype. 
     
                MAPPING: [SRC_LOWER, SRC_UPPER] -> [DEST_LOWER, DEST_UPPER] 
     
              Where this mapping is a linear mapping of values from the left range 
              into the right range of values.  Source pixel values outside the left 
              range are modified to be at the appropriate end of the range.
    """
    pass

def count_points_between_lines(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    count_points_between_lines(*args, **kwargs)
    Overloaded function.
    
    1. count_points_between_lines(l1: dlib.line, l2: dlib.line, reference_point: dlib.dpoint, pts: dlib.points) -> float
    
    2. count_points_between_lines(l1: dlib.line, l2: dlib.line, reference_point: dlib.dpoint, pts: dlib.dpoints) -> float
    
    ensures 
        - Counts and returns the number of points in pts that are between lines l1 and 
          l2.  Since a pair of lines will, in the general case, divide the plane into 4 
          regions, we identify the region of interest as the one that contains the 
          reference_point.  Therefore, this function counts the number of points in pts 
          that appear in the same region as reference_point.
    """
    pass

def count_points_on_side_of_line(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    count_points_on_side_of_line(*args, **kwargs)
    Overloaded function.
    
    1. count_points_on_side_of_line(l: dlib.line, reference_point: dlib.dpoint, pts: dlib.points, dist_thresh_min: float=0, dist_thresh_max: float=inf) -> int
    
    2. count_points_on_side_of_line(l: dlib.line, reference_point: dlib.dpoint, pts: dlib.dpoints, dist_thresh_min: float=0, dist_thresh_max: float=inf) -> int
    
    ensures 
        - Returns a count of how many points in pts have a distance from the line l 
          that is in the range [dist_thresh_min, dist_thresh_max].  This distance is a 
          signed value that indicates how far a point is from the line. Moreover, if 
          the point is on the same side as reference_point then the distance is 
          positive, otherwise it is negative.  So for example, If this range is [0, 
          infinity] then this function counts how many points are on the same side of l 
          as reference_point.
    """
    pass

def count_steps_without_decrease(time_series, probability_of_decrease=0.51): # real signature unknown; restored from __doc__
    """
    count_steps_without_decrease(time_series: object, probability_of_decrease: float=0.51) -> int
    
    requires 
        - time_series must be a one dimensional array of real numbers.  
        - 0.5 < probability_of_decrease < 1 
    ensures 
        - If you think of the contents of time_series as a potentially noisy time 
          series, then this function returns a count of how long the time series has 
          gone without noticeably decreasing in value.  It does this by scanning along 
          the elements, starting from the end (i.e. time_series[-1]) to the beginning, 
          and checking how many elements you need to examine before you are confident 
          that the series has been decreasing in value.  Here, "confident of decrease" 
          means the probability of decrease is >= probability_of_decrease.   
        - Setting probability_of_decrease to 0.51 means we count until we see even a 
          small hint of decrease, whereas a larger value of 0.99 would return a larger 
          count since it keeps going until it is nearly certain the time series is 
          decreasing. 
        - The max possible output from this function is len(time_series). 
        - The implementation of this function is done using the dlib::running_gradient 
          object, which is a tool that finds the least squares fit of a line to the 
          time series and the confidence interval around the slope of that line.  That 
          can then be used in a simple statistical test to determine if the slope is 
          positive or negative.
    """
    return 0

def count_steps_without_decrease_robust(time_series, probability_of_decrease=0.51, quantile_discard=0.1): # real signature unknown; restored from __doc__
    """
    count_steps_without_decrease_robust(time_series: object, probability_of_decrease: float=0.51, quantile_discard: float=0.1) -> int
    
    requires 
        - time_series must be a one dimensional array of real numbers.  
        - 0.5 < probability_of_decrease < 1 
        - 0 <= quantile_discard <= 1 
    ensures 
        - This function behaves just like 
          count_steps_without_decrease(time_series,probability_of_decrease) except that 
          it ignores values in the time series that are in the upper quantile_discard 
          quantile.  So for example, if the quantile discard is 0.1 then the 10% 
          largest values in the time series are ignored.
    """
    return 0

def cross_validate_ranking_trainer(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    cross_validate_ranking_trainer(*args, **kwargs)
    Overloaded function.
    
    1. cross_validate_ranking_trainer(trainer: dlib.svm_rank_trainer, samples: dlib.ranking_pairs, folds: int) -> ranking_test
    
    2. cross_validate_ranking_trainer(trainer: dlib.svm_rank_trainer_sparse, samples: dlib.sparse_ranking_pairs, folds: int) -> ranking_test
    """
    pass

def cross_validate_sequence_segmenter(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    cross_validate_sequence_segmenter(*args, **kwargs)
    Overloaded function.
    
    1. cross_validate_sequence_segmenter(samples: dlib.vectorss, segments: dlib.rangess, folds: int, params: dlib.segmenter_params=<BIO,highFeats,signed,win=5,threads=4,eps=0.1,cache=40,non-verbose,C=100>) -> dlib.segmenter_test
    
    2. cross_validate_sequence_segmenter(samples: dlib.sparse_vectorss, segments: dlib.rangess, folds: int, params: dlib.segmenter_params=<BIO,highFeats,signed,win=5,threads=4,eps=0.1,cache=40,non-verbose,C=100>) -> dlib.segmenter_test
    """
    pass

def cross_validate_trainer(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    cross_validate_trainer(*args, **kwargs)
    Overloaded function.
    
    1. cross_validate_trainer(trainer: dlib.svm_c_trainer_radial_basis, x: dlib.vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    2. cross_validate_trainer(trainer: dlib.svm_c_trainer_sparse_radial_basis, x: dlib.sparse_vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    3. cross_validate_trainer(trainer: dlib.svm_c_trainer_histogram_intersection, x: dlib.vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    4. cross_validate_trainer(trainer: dlib.svm_c_trainer_sparse_histogram_intersection, x: dlib.sparse_vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    5. cross_validate_trainer(trainer: dlib.svm_c_trainer_linear, x: dlib.vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    6. cross_validate_trainer(trainer: dlib.svm_c_trainer_sparse_linear, x: dlib.sparse_vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    7. cross_validate_trainer(trainer: dlib.rvm_trainer_radial_basis, x: dlib.vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    8. cross_validate_trainer(trainer: dlib.rvm_trainer_sparse_radial_basis, x: dlib.sparse_vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    9. cross_validate_trainer(trainer: dlib.rvm_trainer_histogram_intersection, x: dlib.vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    10. cross_validate_trainer(trainer: dlib.rvm_trainer_sparse_histogram_intersection, x: dlib.sparse_vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    11. cross_validate_trainer(trainer: dlib.rvm_trainer_linear, x: dlib.vectors, y: dlib.array, folds: int) -> dlib._binary_test
    
    12. cross_validate_trainer(trainer: dlib.rvm_trainer_sparse_linear, x: dlib.sparse_vectors, y: dlib.array, folds: int) -> dlib._binary_test
    """
    pass

def cross_validate_trainer_threaded(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    cross_validate_trainer_threaded(*args, **kwargs)
    Overloaded function.
    
    1. cross_validate_trainer_threaded(trainer: dlib.svm_c_trainer_radial_basis, x: dlib.vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    2. cross_validate_trainer_threaded(trainer: dlib.svm_c_trainer_sparse_radial_basis, x: dlib.sparse_vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    3. cross_validate_trainer_threaded(trainer: dlib.svm_c_trainer_histogram_intersection, x: dlib.vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    4. cross_validate_trainer_threaded(trainer: dlib.svm_c_trainer_sparse_histogram_intersection, x: dlib.sparse_vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    5. cross_validate_trainer_threaded(trainer: dlib.svm_c_trainer_linear, x: dlib.vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    6. cross_validate_trainer_threaded(trainer: dlib.svm_c_trainer_sparse_linear, x: dlib.sparse_vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    7. cross_validate_trainer_threaded(trainer: dlib.rvm_trainer_radial_basis, x: dlib.vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    8. cross_validate_trainer_threaded(trainer: dlib.rvm_trainer_sparse_radial_basis, x: dlib.sparse_vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    9. cross_validate_trainer_threaded(trainer: dlib.rvm_trainer_histogram_intersection, x: dlib.vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    10. cross_validate_trainer_threaded(trainer: dlib.rvm_trainer_sparse_histogram_intersection, x: dlib.sparse_vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    11. cross_validate_trainer_threaded(trainer: dlib.rvm_trainer_linear, x: dlib.vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    
    12. cross_validate_trainer_threaded(trainer: dlib.rvm_trainer_sparse_linear, x: dlib.sparse_vectors, y: dlib.array, folds: int, num_threads: int) -> dlib._binary_test
    """
    pass

def distance_to_line(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    distance_to_line(*args, **kwargs)
    Overloaded function.
    
    1. distance_to_line(l: dlib.line, p: dlib.point) -> float
    
    2. distance_to_line(l: dlib.line, p: dlib.dpoint) -> float
    
    returns abs(signed_distance_to_line(l,p))
    """
    pass

def dot(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    dot(*args, **kwargs)
    Overloaded function.
    
    1. dot(arg0: dlib.vector, arg1: dlib.vector) -> float
    
    Compute the dot product between two dense column vectors.
    
    2. dot(a: dlib.point, b: dlib.point) -> int
    
    Returns the dot product of the points a and b.
    
    3. dot(a: dlib.dpoint, b: dlib.dpoint) -> float
    
    Returns the dot product of the points a and b.
    """
    pass

def equalize_histogram(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    equalize_histogram(*args, **kwargs)
    Overloaded function.
    
    1. equalize_histogram(img: numpy.ndarray[(rows,cols),uint8]) -> numpy.ndarray[(rows,cols),uint8]
    
    2. equalize_histogram(img: numpy.ndarray[(rows,cols),uint16]) -> numpy.ndarray[(rows,cols),uint16]
    
    Returns a histogram equalized version of img.
    """
    pass

def extract_image_4points(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    extract_image_4points(*args, **kwargs)
    Overloaded function.
    
    1. extract_image_4points(img: numpy.ndarray[(rows,cols),uint8], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),uint8]
    
    2. extract_image_4points(img: numpy.ndarray[(rows,cols),uint16], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),uint16]
    
    3. extract_image_4points(img: numpy.ndarray[(rows,cols),uint32], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),uint32]
    
    4. extract_image_4points(img: numpy.ndarray[(rows,cols),uint64], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),uint64]
    
    5. extract_image_4points(img: numpy.ndarray[(rows,cols),int8], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),int8]
    
    6. extract_image_4points(img: numpy.ndarray[(rows,cols),int16], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),int16]
    
    7. extract_image_4points(img: numpy.ndarray[(rows,cols),int32], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),int32]
    
    8. extract_image_4points(img: numpy.ndarray[(rows,cols),int64], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),int64]
    
    9. extract_image_4points(img: numpy.ndarray[(rows,cols),float32], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),float32]
    
    10. extract_image_4points(img: numpy.ndarray[(rows,cols),float64], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols),float64]
    
    11. extract_image_4points(img: numpy.ndarray[(rows,cols,3),uint8], corners: list, rows: int, columns: int) -> numpy.ndarray[(rows,cols,3),uint8]
    
    requires 
        - corners is a list of dpoint or line objects. 
        - len(corners) == 4 
        - rows >= 0 
        - columns >= 0 
    ensures 
        - The returned image has the given number of rows and columns. 
        - if (corners contains dpoints) then 
            - The 4 points in corners define a convex quadrilateral and this function 
              extracts that part of the input image img and returns it.  Therefore, 
              each corner of the quadrilateral is associated to a corner of the 
              extracted image and bilinear interpolation and a projective mapping is 
              used to transform the pixels in the quadrilateral into the output image. 
              To determine which corners of the quadrilateral map to which corners of 
              the returned image we fit the tightest possible rectangle to the 
              quadrilateral and map its vertices to their nearest rectangle corners. 
              These corners are then trivially mapped to the output image (i.e.  upper 
              left corner to upper left corner, upper right corner to upper right 
              corner, etc.). 
        - else 
            - This routine finds the 4 intersecting points of the given lines which 
              form a convex quadrilateral and uses them as described above to extract 
              an image.   i.e. It just then calls: extract_image_4points(img, 
              intersections_between_lines, rows, columns). 
            - If no convex quadrilateral can be made from the given lines then this 
              routine throws no_convex_quadrilateral.
    """
    pass

def extract_image_chip(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    extract_image_chip(*args, **kwargs)
    Overloaded function.
    
    1. extract_image_chip(img: numpy.ndarray[(rows,cols),uint8], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),uint8]
    
    2. extract_image_chip(img: numpy.ndarray[(rows,cols),uint16], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),uint16]
    
    3. extract_image_chip(img: numpy.ndarray[(rows,cols),uint32], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),uint32]
    
    4. extract_image_chip(img: numpy.ndarray[(rows,cols),uint64], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),uint64]
    
    5. extract_image_chip(img: numpy.ndarray[(rows,cols),int8], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),int8]
    
    6. extract_image_chip(img: numpy.ndarray[(rows,cols),int16], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),int16]
    
    7. extract_image_chip(img: numpy.ndarray[(rows,cols),int32], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),int32]
    
    8. extract_image_chip(img: numpy.ndarray[(rows,cols),int64], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),int64]
    
    9. extract_image_chip(img: numpy.ndarray[(rows,cols),float32], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),float32]
    
    10. extract_image_chip(img: numpy.ndarray[(rows,cols),float64], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols),float64]
    
    11. extract_image_chip(img: numpy.ndarray[(rows,cols,3),uint8], chip_location: dlib.chip_details) -> numpy.ndarray[(rows,cols,3),uint8]
    
        This routine is just like extract_image_chips() except it takes a single 
        chip_details object and returns a single chip image rather than a list of images.
    """
    pass

def extract_image_chips(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    extract_image_chips(*args, **kwargs)
    Overloaded function.
    
    1. extract_image_chips(img: numpy.ndarray[(rows,cols),uint8], chip_locations: list) -> list
    
    2. extract_image_chips(img: numpy.ndarray[(rows,cols),uint16], chip_locations: list) -> list
    
    3. extract_image_chips(img: numpy.ndarray[(rows,cols),uint32], chip_locations: list) -> list
    
    4. extract_image_chips(img: numpy.ndarray[(rows,cols),uint64], chip_locations: list) -> list
    
    5. extract_image_chips(img: numpy.ndarray[(rows,cols),int8], chip_locations: list) -> list
    
    6. extract_image_chips(img: numpy.ndarray[(rows,cols),int16], chip_locations: list) -> list
    
    7. extract_image_chips(img: numpy.ndarray[(rows,cols),int32], chip_locations: list) -> list
    
    8. extract_image_chips(img: numpy.ndarray[(rows,cols),int64], chip_locations: list) -> list
    
    9. extract_image_chips(img: numpy.ndarray[(rows,cols),float32], chip_locations: list) -> list
    
    10. extract_image_chips(img: numpy.ndarray[(rows,cols),float64], chip_locations: list) -> list
    
    11. extract_image_chips(img: numpy.ndarray[(rows,cols,3),uint8], chip_locations: list) -> list
    
    requires 
        - for all valid i:  
            - chip_locations[i].rect.is_empty() == false 
            - chip_locations[i].rows*chip_locations[i].cols != 0 
    ensures 
        - This function extracts "chips" from an image.  That is, it takes a list of 
          rectangular sub-windows (i.e. chips) within an image and extracts those 
          sub-windows, storing each into its own image.  It also scales and rotates the 
          image chips according to the instructions inside each chip_details object. 
          It uses bilinear interpolation. 
        - The extracted image chips are returned in a python list of numpy arrays.  The 
          length of the returned array is len(chip_locations). 
        - Let CHIPS be the returned array, then we have: 
            - for all valid i: 
                - #CHIPS[i] == The image chip extracted from the position 
                  chip_locations[i].rect in img. 
                - #CHIPS[i].shape(0) == chip_locations[i].rows 
                - #CHIPS[i].shape(1) == chip_locations[i].cols 
                - The image will have been rotated counter-clockwise by 
                  chip_locations[i].angle radians, around the center of 
                  chip_locations[i].rect, before the chip was extracted.  
        - Any pixels in an image chip that go outside img are set to 0 (i.e. black).
    """
    pass

def find_bright_keypoints(xx, float32=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    find_bright_keypoints(xx: numpy.ndarray[(rows,cols),float32], xy: numpy.ndarray[(rows,cols),float32], yy: numpy.ndarray[(rows,cols),float32]) -> numpy.ndarray[(rows,cols),float32]
    
    requires 
        - xx, xy, and yy all have the same dimensions. 
    ensures 
        - This routine finds bright "keypoints" in an image.  In general, these are 
          bright/white localized blobs.  It does this by computing the determinant of 
          the image Hessian at each location and storing this value into the returned 
          image if both eigenvalues of the Hessian are negative.  If either eigenvalue 
          is positive then the output value for that pixel is 0.  I.e. 
            - Let OUT denote the returned image. 
            - for all valid r,c: 
                - OUT[r][c] == a number >= 0 and larger values indicate the 
                  presence of a keypoint at this pixel location. 
        - We assume that xx, xy, and yy are the 3 second order gradients of the image 
          in question.  You can obtain these gradients using the image_gradients class. 
        - The output image will have the same dimensions as the input images.
    """
    pass

def find_bright_lines(xx, float32=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    find_bright_lines(xx: numpy.ndarray[(rows,cols),float32], xy: numpy.ndarray[(rows,cols),float32], yy: numpy.ndarray[(rows,cols),float32]) -> tuple
    
    requires 
        - xx, xy, and yy all have the same dimensions. 
    ensures 
        - This routine is similar to sobel_edge_detector(), except instead of finding 
          an edge it finds a bright/white line.  For example, the border between a 
          black piece of paper and a white table is an edge, but a curve drawn with a 
          pencil on a piece of paper makes a line.  Therefore, the output of this 
          routine is a vector field encoded in the horz and vert images, which are 
          returned in a tuple where the first element is horz and the second is vert. 
     
          The vector obtains a large magnitude when centered on a bright line in an image and the 
          direction of the vector is perpendicular to the line.  To be very precise, 
          each vector points in the direction of greatest change in second derivative 
          and the magnitude of the vector encodes the derivative magnitude in that 
          direction.  Moreover, if the second derivative is positive then the output 
          vector is zero.  This zeroing if positive gradients causes the output to be 
          sensitive only to bright lines surrounded by darker pixels. 
     
        - We assume that xx, xy, and yy are the 3 second order gradients of the image 
          in question.  You can obtain these gradients using the image_gradients class. 
        - The output images will have the same dimensions as the input images.
    """
    pass

def find_candidate_object_locations(image, rects, kvals=502003, min_size=20, max_merging_iterations=50): # real signature unknown; restored from __doc__
    """
    find_candidate_object_locations(image: array, rects: list, kvals: tuple=(50, 200, 3), min_size: int=20, max_merging_iterations: int=50) -> None
    
    Returns found candidate objects
    requires
        - image == an image object which is a numpy ndarray
        - len(kvals) == 3
        - kvals should be a tuple that specifies the range of k values to use.  In
          particular, it should take the form (start, end, num) where num > 0. 
    ensures
        - This function takes an input image and generates a set of candidate
          rectangles which are expected to bound any objects in the image.  It does
          this by running a version of the segment_image() routine on the image and
          then reports rectangles containing each of the segments as well as rectangles
          containing unions of adjacent segments.  The basic idea is described in the
          paper: 
              Segmentation as Selective Search for Object Recognition by Koen E. A. van de Sande, et al.
          Note that this function deviates from what is described in the paper slightly. 
          See the code for details.
        - The basic segmentation is performed kvals[2] times, each time with the k parameter
          (see segment_image() and the Felzenszwalb paper for details on k) set to a different
          value from the range of numbers linearly spaced between kvals[0] to kvals[1].
        - When doing the basic segmentations prior to any box merging, we discard all
          rectangles that have an area < min_size.  Therefore, all outputs and
          subsequent merged rectangles are built out of rectangles that contain at
          least min_size pixels.  Note that setting min_size to a smaller value than
          you might otherwise be interested in using can be useful since it allows a
          larger number of possible merged boxes to be created.
        - There are max_merging_iterations rounds of neighboring blob merging.
          Therefore, this parameter has some effect on the number of output rectangles
          you get, with larger values of the parameter giving more output rectangles.
        - This function appends the output rectangles into #rects.  This means that any
          rectangles in rects before this function was called will still be in there
          after it terminates.  Note further that #rects will not contain any duplicate
          rectangles.  That is, for all valid i and j where i != j it will be true
          that:
            - #rects[i] != rects[j]
    """
    pass

def find_dark_keypoints(xx, float32=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    find_dark_keypoints(xx: numpy.ndarray[(rows,cols),float32], xy: numpy.ndarray[(rows,cols),float32], yy: numpy.ndarray[(rows,cols),float32]) -> numpy.ndarray[(rows,cols),float32]
    
    requires 
        - xx, xy, and yy all have the same dimensions. 
    ensures 
        - This routine finds dark "keypoints" in an image.  In general, these are 
          dark localized blobs.  It does this by computing the determinant of 
          the image Hessian at each location and storing this value into the returned 
          image if both eigenvalues of the Hessian are negative.  If either eigenvalue 
          is negative then the output value for that pixel is 0.  I.e. 
            - Let OUT denote the returned image. 
            - for all valid r,c: 
                - OUT[r][c] == a number >= 0 and larger values indicate the 
                  presence of a keypoint at this pixel location. 
        - We assume that xx, xy, and yy are the 3 second order gradients of the image 
          in question.  You can obtain these gradients using the image_gradients class. 
        - The output image will have the same dimensions as the input images.
    """
    pass

def find_dark_lines(xx, float32=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    find_dark_lines(xx: numpy.ndarray[(rows,cols),float32], xy: numpy.ndarray[(rows,cols),float32], yy: numpy.ndarray[(rows,cols),float32]) -> tuple
    
    requires 
        - xx, xy, and yy all have the same dimensions. 
    ensures 
        - This routine is similar to sobel_edge_detector(), except instead of finding 
          an edge it finds a dark line.  For example, the border between a black piece 
          of paper and a white table is an edge, but a curve drawn with a pencil on a 
          piece of paper makes a line.  Therefore, the output of this routine is a 
          vector field encoded in the horz and vert images, which are returned in a 
          tuple where the first element is horz and the second is vert. 
     
          The vector obtains a large magnitude when centered on a dark line in an image 
          and the direction of the vector is perpendicular to the line.  To be very 
          precise, each vector points in the direction of greatest change in second 
          derivative and the magnitude of the vector encodes the derivative magnitude 
          in that direction.  Moreover, if the second derivative is negative then the 
          output vector is zero.  This zeroing if negative gradients causes the output 
          to be sensitive only to dark lines surrounded by darker pixels. 
     
        - We assume that xx, xy, and yy are the 3 second order gradients of the image 
          in question.  You can obtain these gradients using the image_gradients class. 
        - The output images will have the same dimensions as the input images.
    """
    pass

def find_line_endpoints(img, uint8=None): # real signature unknown; restored from __doc__
    """
    find_line_endpoints(img: numpy.ndarray[(rows,cols),uint8]) -> dlib.points
    
    requires 
        - all pixels in img are set to either 255 or 0. 
          (i.e. it must be a binary image) 
    ensures 
        - This routine finds endpoints of lines in a thinned binary image.  For 
          example, if the image was produced by skeleton() or something like a Canny 
          edge detector then you can use find_line_endpoints() to find the pixels 
          sitting on the ends of lines.
    """
    pass

def find_max_global(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    find_max_global(*args, **kwargs)
    Overloaded function.
    
    1. find_max_global(f: object, bound1: list, bound2: list, is_integer_variable: list, num_function_calls: int, solver_epsilon: float=0) -> tuple
    
    requires 
        - len(bound1) == len(bound2) == len(is_integer_variable) 
        - for all valid i: bound1[i] != bound2[i] 
        - solver_epsilon >= 0 
        - f() is a real valued multi-variate function.  It must take scalar real 
          numbers as its arguments and the number of arguments must be len(bound1). 
    ensures 
        - This function performs global optimization on the given f() function. 
          The goal is to maximize the following objective function: 
             f(x) 
          subject to the constraints: 
            min(bound1[i],bound2[i]) <= x[i] <= max(bound1[i],bound2[i]) 
            if (is_integer_variable[i]) then x[i] is an integer value (but still 
            represented with float type). 
        - find_max_global() runs until it has called f() num_function_calls times. 
          Then it returns the best x it has found along with the corresponding output 
          of f().  That is, it returns (best_x_seen,f(best_x_seen)).  Here best_x_seen 
          is a list containing the best arguments to f() this function has found. 
        - find_max_global() uses a global optimization method based on a combination of 
          non-parametric global function modeling and quadratic trust region modeling 
          to efficiently find a global maximizer.  It usually does a good job with a 
          relatively small number of calls to f().  For more information on how it 
          works read the documentation for dlib's global_function_search object. 
          However, one notable element is the solver epsilon, which you can adjust. 
     
          The search procedure will only attempt to find a global maximizer to at most 
          solver_epsilon accuracy.  Once a local maximizer is found to that accuracy 
          the search will focus entirely on finding other maxima elsewhere rather than 
          on further improving the current local optima found so far.  That is, once a 
          local maxima is identified to about solver_epsilon accuracy, the algorithm 
          will spend all its time exploring the function to find other local maxima to 
          investigate.  An epsilon of 0 means it will keep solving until it reaches 
          full floating point precision.  Larger values will cause it to switch to pure 
          global exploration sooner and therefore might be more effective if your 
          objective function has many local maxima and you don't care about a super 
          high precision solution. 
        - Any variables that satisfy the following conditions are optimized on a log-scale: 
            - The lower bound on the variable is > 0 
            - The ratio of the upper bound to lower bound is > 1000 
            - The variable is not an integer variable 
          We do this because it's common to optimize machine learning models that have 
          parameters with bounds in a range such as [1e-5 to 1e10] (e.g. the SVM C 
          parameter) and it's much more appropriate to optimize these kinds of 
          variables on a log scale.  So we transform them by applying log() to 
          them and then undo the transform via exp() before invoking the function 
          being optimized.  Therefore, this transformation is invisible to the user 
          supplied functions.  In most cases, it improves the efficiency of the 
          optimizer.
    
    2. find_max_global(f: object, bound1: list, bound2: list, num_function_calls: int, solver_epsilon: float=0) -> tuple
    
    This function simply calls the other version of find_max_global() with is_integer_variable set to False for all variables.
    """
    pass

def find_min_global(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    find_min_global(*args, **kwargs)
    Overloaded function.
    
    1. find_min_global(f: object, bound1: list, bound2: list, is_integer_variable: list, num_function_calls: int, solver_epsilon: float=0) -> tuple
    
    This function is just like find_max_global(), except it performs minimization rather than maximization.
    
    2. find_min_global(f: object, bound1: list, bound2: list, num_function_calls: int, solver_epsilon: float=0) -> tuple
    
    This function simply calls the other version of find_min_global() with is_integer_variable set to False for all variables.
    """
    pass

def find_optimal_rect_filter(rects, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    find_optimal_rect_filter(rects: std::__1::vector<dlib::rectangle, std::__1::allocator<dlib::rectangle> >, smoothness: float=1) -> dlib.rect_filter
    
    requires 
        - rects.size() > 4 
        - smoothness >= 0 
    ensures 
        - This function finds the "optimal" settings of a rect_filter based on recorded 
          measurement data stored in rects.  Here we assume that rects is a complete 
          track history of some object's measured positions.  Essentially, what we do 
          is find the rect_filter that minimizes the following objective function: 
             sum of abs(predicted_location[i] - measured_location[i]) + smoothness*abs(filtered_location[i]-filtered_location[i-1]) 
             Where i is a time index. 
          The sum runs over all the data in rects.  So what we do is find the 
          filter settings that produce smooth filtered trajectories but also produce 
          filtered outputs that are as close to the measured positions as possible. 
          The larger the value of smoothness the less jittery the filter outputs will 
          be, but they might become biased or laggy if smoothness is set really high.
    """
    pass

def find_peaks(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    find_peaks(*args, **kwargs)
    Overloaded function.
    
    1. find_peaks(img: numpy.ndarray[(rows,cols),float32], non_max_suppression_radius: float, thresh: float) -> dlib.points
    
    2. find_peaks(img: numpy.ndarray[(rows,cols),float64], non_max_suppression_radius: float, thresh: float) -> dlib.points
    
    3. find_peaks(img: numpy.ndarray[(rows,cols),uint8], non_max_suppression_radius: float, thresh: int) -> dlib.points
    
    4. find_peaks(img: numpy.ndarray[(rows,cols),uint16], non_max_suppression_radius: float, thresh: int) -> dlib.points
    
    5. find_peaks(img: numpy.ndarray[(rows,cols),uint32], non_max_suppression_radius: float, thresh: int) -> dlib.points
    
    6. find_peaks(img: numpy.ndarray[(rows,cols),uint64], non_max_suppression_radius: float, thresh: int) -> dlib.points
    
    7. find_peaks(img: numpy.ndarray[(rows,cols),int8], non_max_suppression_radius: float, thresh: int) -> dlib.points
    
    8. find_peaks(img: numpy.ndarray[(rows,cols),int16], non_max_suppression_radius: float, thresh: int) -> dlib.points
    
    9. find_peaks(img: numpy.ndarray[(rows,cols),int32], non_max_suppression_radius: float, thresh: int) -> dlib.points
    
    10. find_peaks(img: numpy.ndarray[(rows,cols),int64], non_max_suppression_radius: float, thresh: int) -> dlib.points
    
    requires 
        - non_max_suppression_radius >= 0 
    ensures 
        - Scans the given image and finds all pixels with values >= thresh that are 
          also local maximums within their 8-connected neighborhood of the image.  Such 
          pixels are collected, sorted in decreasing order of their pixel values, and 
          then non-maximum suppression is applied to this list of points using the 
          given non_max_suppression_radius.  The final list of peaks is then returned. 
     
          Therefore, the returned list, V, will have these properties: 
            - len(V) == the number of peaks found in the image. 
            - When measured in image coordinates, no elements of V are within 
              non_max_suppression_radius distance of each other.  That is, for all valid i!=j 
              it is true that length(V[i]-V[j]) > non_max_suppression_radius. 
            - For each element of V, that element has the maximum pixel value of all 
              pixels in the ball centered on that pixel with radius 
              non_max_suppression_radius.
    
    11. find_peaks(img: numpy.ndarray[(rows,cols),float32], non_max_suppression_radius: float=0) -> dlib.points
    
    12. find_peaks(img: numpy.ndarray[(rows,cols),float64], non_max_suppression_radius: float=0) -> dlib.points
    
    13. find_peaks(img: numpy.ndarray[(rows,cols),uint8], non_max_suppression_radius: float=0) -> dlib.points
    
    14. find_peaks(img: numpy.ndarray[(rows,cols),uint16], non_max_suppression_radius: float=0) -> dlib.points
    
    15. find_peaks(img: numpy.ndarray[(rows,cols),uint32], non_max_suppression_radius: float=0) -> dlib.points
    
    16. find_peaks(img: numpy.ndarray[(rows,cols),uint64], non_max_suppression_radius: float=0) -> dlib.points
    
    17. find_peaks(img: numpy.ndarray[(rows,cols),int8], non_max_suppression_radius: float=0) -> dlib.points
    
    18. find_peaks(img: numpy.ndarray[(rows,cols),int16], non_max_suppression_radius: float=0) -> dlib.points
    
    19. find_peaks(img: numpy.ndarray[(rows,cols),int32], non_max_suppression_radius: float=0) -> dlib.points
    
    20. find_peaks(img: numpy.ndarray[(rows,cols),int64], non_max_suppression_radius: float=0) -> dlib.points
    
    performs: return find_peaks(img, non_max_suppression_radius, partition_pixels(img))
    """
    pass

def find_projective_transform(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    find_projective_transform(*args, **kwargs)
    Overloaded function.
    
    1. find_projective_transform(from_points: dlib.dpoints, to_points: dlib.dpoints) -> dlib.point_transform_projective
    
    requires 
        - len(from_points) == len(to_points) 
        - len(from_points) >= 4 
    ensures 
        - returns a point_transform_projective object, T, such that for all valid i: 
            length(T(from_points[i]) - to_points[i]) 
          is minimized as often as possible.  That is, this function finds the projective 
          transform that maps points in from_points to points in to_points.  If no 
          projective transform exists which performs this mapping exactly then the one 
          which minimizes the mean squared error is selected. 
    
    2. find_projective_transform(from_points: numpy.ndarray[(rows,cols),float32], to_points: numpy.ndarray[(rows,cols),float32]) -> dlib.point_transform_projective
    
    requires 
        - from_points and to_points have two columns and the same number of rows. 
          Moreover, they have at least 4 rows. 
    ensures 
        - returns a point_transform_projective object, T, such that for all valid i: 
            length(T(dpoint(from_points[i])) - dpoint(to_points[i])) 
          is minimized as often as possible.  That is, this function finds the projective 
          transform that maps points in from_points to points in to_points.  If no 
          projective transform exists which performs this mapping exactly then the one 
          which minimizes the mean squared error is selected. 
    
    3. find_projective_transform(from_points: numpy.ndarray[(rows,cols),float64], to_points: numpy.ndarray[(rows,cols),float64]) -> dlib.point_transform_projective
    
    requires 
        - from_points and to_points have two columns and the same number of rows. 
          Moreover, they have at least 4 rows. 
    ensures 
        - returns a point_transform_projective object, T, such that for all valid i: 
            length(T(dpoint(from_points[i])) - dpoint(to_points[i])) 
          is minimized as often as possible.  That is, this function finds the projective 
          transform that maps points in from_points to points in to_points.  If no 
          projective transform exists which performs this mapping exactly then the one 
          which minimizes the mean squared error is selected.
    """
    pass

def gaussian_blur(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    gaussian_blur(*args, **kwargs)
    Overloaded function.
    
    1. gaussian_blur(img: numpy.ndarray[(rows,cols,3),uint8], sigma: float, max_size: int=1000) -> tuple
    
    2. gaussian_blur(img: numpy.ndarray[(rows,cols),uint8], sigma: float, max_size: int=1000) -> tuple
    
    3. gaussian_blur(img: numpy.ndarray[(rows,cols),uint16], sigma: float, max_size: int=1000) -> tuple
    
    4. gaussian_blur(img: numpy.ndarray[(rows,cols),uint32], sigma: float, max_size: int=1000) -> tuple
    
    5. gaussian_blur(img: numpy.ndarray[(rows,cols),float32], sigma: float, max_size: int=1000) -> tuple
    
    6. gaussian_blur(img: numpy.ndarray[(rows,cols),float64], sigma: float, max_size: int=1000) -> tuple
    
    requires 
        - sigma > 0 
        - max_size > 0 
        - max_size is an odd number 
    ensures 
        - Filters img with a Gaussian filter of sigma width.  The actual spatial filter will 
          be applied to pixel blocks that are at most max_size wide and max_size tall (note that 
          this function will automatically select a smaller block size as appropriate).  The  
          results are returned.  We also return a rectangle which indicates what pixels 
          in the returned image are considered non-border pixels and therefore contain 
          output from the filter.  E.g. 
            - filtered_img,rect = gaussian_blur(img) 
          would give you the filtered image and the rectangle in question. 
        - The filter is applied to each color channel independently. 
        - Pixels close enough to the edge of img to not have the filter still fit  
          inside the image are set to zero. 
        - The returned image has the same dimensions as the input image.
    """
    pass

def get_face_chip(img, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    get_face_chip(img: numpy.ndarray[(rows,cols,3),uint8], face: dlib.full_object_detection, size: int=150, padding: float=0.25) -> numpy.ndarray[(rows,cols,3),uint8]
    
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
    pass

def get_face_chips(img, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    get_face_chips(img: numpy.ndarray[(rows,cols,3),uint8], faces: dlib.full_object_detections, size: int=150, padding: float=0.25) -> list
    
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
    pass

def get_frontal_face_detector(): # real signature unknown; restored from __doc__
    """
    get_frontal_face_detector() -> dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6u>, dlib::default_fhog_feature_extractor> >
    
    Returns the default face detector
    """
    pass

def get_histogram(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    get_histogram(*args, **kwargs)
    Overloaded function.
    
    1. get_histogram(img: numpy.ndarray[(rows,cols),uint8], hist_size: int) -> numpy.ndarray[uint64]
    
    2. get_histogram(img: numpy.ndarray[(rows,cols),uint16], hist_size: int) -> numpy.ndarray[uint64]
    
    3. get_histogram(img: numpy.ndarray[(rows,cols),uint32], hist_size: int) -> numpy.ndarray[uint64]
    
    4. get_histogram(img: numpy.ndarray[(rows,cols),uint64], hist_size: int) -> numpy.ndarray[uint64]
    
    ensures 
        - Returns a numpy array, HIST, that contains a histogram of the pixels in img. 
          In particular, we will have: 
            - len(HIST) == hist_size 
            - for all valid i:  
                - HIST[i] == the number of times a pixel with intensity i appears in img.
    """
    pass

def get_rect(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    get_rect(*args, **kwargs)
    Overloaded function.
    
    1. get_rect(img: array) -> dlib.rectangle
    
    returns a rectangle(0,0,img.shape(1)-1,img.shape(0)-1).  Therefore, it is the rectangle that bounds the image.
    
    2. get_rect(ht: dlib.hough_transform) -> dlib.rectangle
    
    returns a rectangle(0,0,ht.size()-1,ht.size()-1).  Therefore, it is the rectangle that bounds the Hough transform image.
    """
    pass

def grow_rect(rect, num): # real signature unknown; restored from __doc__
    """
    grow_rect(rect: dlib.rectangle, num: int) -> dlib.rectangle
    
    - return shrink_rect(rect, -num) 
      (i.e. grows the given rectangle by expanding its border by num)
    """
    pass

def hit_enter_to_continue(): # real signature unknown; restored from __doc__
    """
    hit_enter_to_continue() -> None
    
    Asks the user to hit enter to continue and pauses until they do so.
    """
    pass

def hysteresis_threshold(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    hysteresis_threshold(*args, **kwargs)
    Overloaded function.
    
    1. hysteresis_threshold(img: numpy.ndarray[(rows,cols),uint8], lower_thresh: int, upper_thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    2. hysteresis_threshold(img: numpy.ndarray[(rows,cols),uint16], lower_thresh: int, upper_thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    3. hysteresis_threshold(img: numpy.ndarray[(rows,cols),uint32], lower_thresh: int, upper_thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    4. hysteresis_threshold(img: numpy.ndarray[(rows,cols),uint64], lower_thresh: int, upper_thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    5. hysteresis_threshold(img: numpy.ndarray[(rows,cols),int8], lower_thresh: int, upper_thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    6. hysteresis_threshold(img: numpy.ndarray[(rows,cols),int16], lower_thresh: int, upper_thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    7. hysteresis_threshold(img: numpy.ndarray[(rows,cols),int32], lower_thresh: int, upper_thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    8. hysteresis_threshold(img: numpy.ndarray[(rows,cols),int64], lower_thresh: int, upper_thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    9. hysteresis_threshold(img: numpy.ndarray[(rows,cols),float32], lower_thresh: float, upper_thresh: float) -> numpy.ndarray[(rows,cols),uint8]
    
    10. hysteresis_threshold(img: numpy.ndarray[(rows,cols),float64], lower_thresh: float, upper_thresh: float) -> numpy.ndarray[(rows,cols),uint8]
    
    Applies hysteresis thresholding to img and returns the results.  In particular, 
    pixels in img with values >= upper_thresh have an output value of 255 and all 
    others have a value of 0 unless they are >= lower_thresh and are connected to a 
    pixel with a value >= upper_thresh, in which case they have a value of 255.  Here 
    pixels are connected if there is a path between them composed of pixels that would 
    receive an output of 255.
    
    11. hysteresis_threshold(img: numpy.ndarray[(rows,cols),uint8]) -> numpy.ndarray[(rows,cols),uint8]
    
    12. hysteresis_threshold(img: numpy.ndarray[(rows,cols),uint16]) -> numpy.ndarray[(rows,cols),uint8]
    
    13. hysteresis_threshold(img: numpy.ndarray[(rows,cols),uint32]) -> numpy.ndarray[(rows,cols),uint8]
    
    14. hysteresis_threshold(img: numpy.ndarray[(rows,cols),uint64]) -> numpy.ndarray[(rows,cols),uint8]
    
    15. hysteresis_threshold(img: numpy.ndarray[(rows,cols),int8]) -> numpy.ndarray[(rows,cols),uint8]
    
    16. hysteresis_threshold(img: numpy.ndarray[(rows,cols),int16]) -> numpy.ndarray[(rows,cols),uint8]
    
    17. hysteresis_threshold(img: numpy.ndarray[(rows,cols),int32]) -> numpy.ndarray[(rows,cols),uint8]
    
    18. hysteresis_threshold(img: numpy.ndarray[(rows,cols),int64]) -> numpy.ndarray[(rows,cols),uint8]
    
    19. hysteresis_threshold(img: numpy.ndarray[(rows,cols),float32]) -> numpy.ndarray[(rows,cols),uint8]
    
    20. hysteresis_threshold(img: numpy.ndarray[(rows,cols),float64]) -> numpy.ndarray[(rows,cols),uint8]
    
    performs: return hysteresis_threshold(img, t1, t2) where the thresholds 
    are first obtained by calling [t1, t2]=partition_pixels(img).
    """
    pass

def intersect(a, b): # real signature unknown; restored from __doc__
    """
    intersect(a: dlib.line, b: dlib.line) -> dlib.dpoint
    
    ensures 
        - returns the point of intersection between lines a and b.  If no such point 
          exists then this function returns a point with Inf values in it.
    """
    pass

def inv(trans): # real signature unknown; restored from __doc__
    """
    inv(trans: dlib.point_transform_projective) -> dlib.point_transform_projective
    
    ensures 
        - If trans is an invertible transformation then this function returns a new 
          transformation that is the inverse of trans.
    """
    pass

def jet(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    jet(*args, **kwargs)
    Overloaded function.
    
    1. jet(img: numpy.ndarray[(rows,cols),uint8]) -> numpy.ndarray[(rows,cols,3),uint8]
    
    2. jet(img: numpy.ndarray[(rows,cols),uint16]) -> numpy.ndarray[(rows,cols,3),uint8]
    
    3. jet(img: numpy.ndarray[(rows,cols),uint32]) -> numpy.ndarray[(rows,cols,3),uint8]
    
    4. jet(img: numpy.ndarray[(rows,cols),float32]) -> numpy.ndarray[(rows,cols,3),uint8]
    
    5. jet(img: numpy.ndarray[(rows,cols),float64]) -> numpy.ndarray[(rows,cols,3),uint8]
    
    Converts a grayscale image into a jet colored image.  This is an image where dark 
    pixels are dark blue and larger values become light blue, then yellow, and then 
    finally red as they approach the maximum pixel values.
    """
    pass

def jitter_image(img, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    jitter_image(img: numpy.ndarray[(rows,cols,3),uint8], num_jitters: int=1, disturb_colors: bool=False) -> list
    
    Takes an image and returns a list of jittered images.The returned list contains num_jitters images (default is 1).If disturb_colors is set to True, the colors of the image are disturbed (default is False)
    """
    pass

def label_connected_blobs(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    label_connected_blobs(*args, **kwargs)
    Overloaded function.
    
    1. label_connected_blobs(img: numpy.ndarray[(rows,cols),uint8], zero_pixels_are_background: bool=True, neighborhood_connectivity: int=8, connected_if_both_not_zero: bool=False) -> tuple
    
    2. label_connected_blobs(img: numpy.ndarray[(rows,cols),uint16], zero_pixels_are_background: bool=True, neighborhood_connectivity: int=8, connected_if_both_not_zero: bool=False) -> tuple
    
    3. label_connected_blobs(img: numpy.ndarray[(rows,cols),uint32], zero_pixels_are_background: bool=True, neighborhood_connectivity: int=8, connected_if_both_not_zero: bool=False) -> tuple
    
    4. label_connected_blobs(img: numpy.ndarray[(rows,cols),uint64], zero_pixels_are_background: bool=True, neighborhood_connectivity: int=8, connected_if_both_not_zero: bool=False) -> tuple
    
    5. label_connected_blobs(img: numpy.ndarray[(rows,cols),float32], zero_pixels_are_background: bool=True, neighborhood_connectivity: int=8, connected_if_both_not_zero: bool=False) -> tuple
    
    6. label_connected_blobs(img: numpy.ndarray[(rows,cols),float64], zero_pixels_are_background: bool=True, neighborhood_connectivity: int=8, connected_if_both_not_zero: bool=False) -> tuple
    
    requires 
        - neighborhood_connectivity == 4, 8, or 24 
    ensures 
        - This function labels each of the connected blobs in img with a unique integer  
          label.   
        - An image can be thought of as a graph where pixels A and B are connected if 
          they are close to each other and satisfy some criterion like having the same 
          value or both being non-zero.  Then this function can be understood as 
          labeling all the connected components of this pixel graph such that all 
          pixels in a component get the same label while pixels in different components 
          get different labels.   
        - If zero_pixels_are_background==true then there is a special background component 
          and all pixels with value 0 are assigned to it. Moreover, all such background pixels 
          will always get a blob id of 0 regardless of any other considerations. 
        - This function returns a label image and a count of the number of blobs found. 
          I.e., if you ran this function like: 
            label_img, num_blobs = label_connected_blobs(img) 
          You would obtain the noted label image and number of blobs. 
        - The output label_img has the same dimensions as the input image. 
        - for all valid r and c: 
            - label_img[r][c] == the blob label number for pixel img[r][c].   
            - label_img[r][c] >= 0 
            - if (img[r][c]==0) then 
                - label_img[r][c] == 0 
            - else 
                - label_img[r][c] != 0 
        - if (len(img) != 0) then  
            - The returned num_blobs will be == label_img.max()+1 
              (i.e. returns a number one greater than the maximum blob id number,  
              this is the number of blobs found.) 
        - else 
            - num_blobs will be 0. 
        - blob labels are contiguous, therefore, the number returned by this function is 
          the number of blobs in the image (including the background blob).
    """
    pass

def label_connected_blobs_watershed(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    label_connected_blobs_watershed(*args, **kwargs)
    Overloaded function.
    
    1. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),uint8], background_thresh: int, smoothing: float=0) -> tuple
    
    2. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),uint16], background_thresh: int, smoothing: float=0) -> tuple
    
    3. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),uint32], background_thresh: int, smoothing: float=0) -> tuple
    
    4. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),float32], background_thresh: float, smoothing: float=0) -> tuple
    
    5. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),float64], background_thresh: float, smoothing: float=0) -> tuple
    
    requires 
        - smoothing >= 0 
    ensures 
        - This routine performs a watershed segmentation of the given input image and 
          labels each resulting flooding region with a unique integer label. It does 
          this by marking the brightest pixels as sources of flooding and then flood 
          fills the image outward from those sources.  Each flooded area is labeled 
          with the identity of the source pixel and flooding stops when another flooded 
          area is reached or pixels with values < background_thresh are encountered.   
        - The flooding will also overrun a source pixel if that source pixel has yet to 
          label any neighboring pixels.  This behavior helps to mitigate spurious 
          splits of objects due to noise.  You can further control this behavior by 
          setting the smoothing parameter.  The flooding will take place on an image 
          that has been Gaussian blurred with a sigma==smoothing.  So setting smoothing 
          to a larger number will in general cause more regions to be merged together. 
          Note that the smoothing parameter has no effect on the interpretation of 
          background_thresh since the decision of "background or not background" is 
          always made relative to the unsmoothed input image. 
        - This function returns a tuple of the labeled image and number of blobs found.  
          i.e. you can call it like this: 
            label_img, num_blobs = label_connected_blobs_watershed(img,background_thresh,smoothing) 
        - The returned label_img will have the same dimensions as img.  
        - for all valid r and c: 
            - if (img[r][c] < background_thresh) then 
                - label_img[r][c] == 0, (i.e. the pixel is labeled as background) 
            - else 
                - label_img[r][c] == an integer value indicating the identity of the segment 
                  containing the pixel img[r][c].   
        - The returned num_blobs is the number of labeled segments, including the 
          background segment.  Therefore, the returned number is 1+(the max value in 
          label_img).
    
    6. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),uint8]) -> tuple
    
    7. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),uint16]) -> tuple
    
    8. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),uint32]) -> tuple
    
    9. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),float32]) -> tuple
    
    10. label_connected_blobs_watershed(img: numpy.ndarray[(rows,cols),float64]) -> tuple
    
    This version of label_connected_blobs_watershed simple invokes: 
       return label_connected_blobs_watershed(img, partition_pixels(img))
    """
    pass

def length(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    length(*args, **kwargs)
    Overloaded function.
    
    1. length(p: dlib.point) -> float
    
    returns the distance from p to the origin, i.e. the L2 norm of p.
    
    2. length(p: dlib.dpoint) -> float
    
    returns the distance from p to the origin, i.e. the L2 norm of p.
    """
    pass

def load_grayscale_image(filename): # real signature unknown; restored from __doc__
    """
    load_grayscale_image(filename: str) -> numpy.ndarray[(rows,cols),uint8]
    
    Takes a path and returns a numpy array containing the image, as an 8bit grayscale image.
    """
    pass

def load_libsvm_formatted_data(file_name): # real signature unknown; restored from __doc__
    """
    load_libsvm_formatted_data(file_name: str) -> tuple
    
    ensures    
        - Attempts to read a file of the given name that should contain libsvm    
          formatted data.  The data is returned as a tuple where the first tuple    
          element is an array of sparse vectors and the second element is an array of    
          labels.
    """
    return ()

def load_rgb_image(filename): # real signature unknown; restored from __doc__
    """
    load_rgb_image(filename: str) -> numpy.ndarray[(rows,cols,3),uint8]
    
    Takes a path and returns a numpy array (RGB) containing the image
    """
    pass

def make_bounding_box_regression_training_data(truth, detections): # real signature unknown; restored from __doc__
    """
    make_bounding_box_regression_training_data(truth: dlib.image_dataset_metadata.dataset, detections: object) -> dlib.image_dataset_metadata.dataset
    
    requires 
        - len(truth.images) == len(detections) 
        - detections == A dlib.rectangless object or a list of dlib.rectangles. 
    ensures 
        - Suppose you have an object detector that can roughly locate objects in an 
          image.  This means your detector draws boxes around objects, but these are 
          *rough* boxes in the sense that they aren't positioned super accurately.  For 
          instance, HOG based detectors usually have a stride of 8 pixels.  So the 
          positional accuracy is going to be, at best, +/-8 pixels.   
           
          If you want to get better positional accuracy one easy thing to do is train a 
          shape_predictor to give you the corners of the object.  The 
          make_bounding_box_regression_training_data() routine helps you do this by 
          creating an appropriate training dataset.  It does this by taking the dataset 
          you used to train your detector (the truth object), and combining that with 
          the output of your detector on each image in the training dataset (the 
          detections object).  In particular, it will create a new annotated dataset 
          where each object box is one of the rectangles from detections and that 
          object has 4 part annotations, the corners of the truth rectangle 
          corresponding to that detection rectangle.  You can then take the returned 
          dataset and train a shape_predictor on it.  The resulting shape_predictor can 
          then be used to do bounding box regression. 
        - We assume that detections[i] contains object detections corresponding to  
          the image truth.images[i].
    """
    pass

def make_sparse_vector(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    make_sparse_vector(*args, **kwargs)
    Overloaded function.
    
    1. make_sparse_vector(arg0: dlib.sparse_vector) -> None
    
    This function modifies its argument so that it is a properly sorted sparse vector.    
    This means that the elements of the sparse vector will be ordered so that pairs    
    with smaller indices come first.  Additionally, there won't be any pairs with    
    identical indices.  If such pairs were present in the input sparse vector then    
    their values will be added together and only one pair with their index will be    
    present in the output.   
    
    2. make_sparse_vector(arg0: dlib.sparse_vectors) -> None
    
    This function modifies a sparse_vectors object so that all elements it contains are properly sorted sparse vectors.
    """
    pass

def max_cost_assignment(cost): # real signature unknown; restored from __doc__
    """
    max_cost_assignment(cost: dlib.matrix) -> list
    
    requires    
        - cost.nr() == cost.nc()    
          (i.e. the input must be a square matrix)    
    ensures    
        - Finds and returns the solution to the following optimization problem:    
        
            Maximize: f(A) == assignment_cost(cost, A)    
            Subject to the following constraints:    
                - The elements of A are unique. That is, there aren't any     
                  elements of A which are equal.      
                - len(A) == cost.nr()    
        
        - Note that this function converts the input cost matrix into a 64bit fixed    
          point representation.  Therefore, you should make sure that the values in    
          your cost matrix can be accurately represented by 64bit fixed point values.    
          If this is not the case then the solution my become inaccurate due to    
          rounding error.  In general, this function will work properly when the ratio    
          of the largest to the smallest value in cost is no more than about 1e16.
    """
    return []

def max_index_plus_one(v): # real signature unknown; restored from __doc__
    """
    max_index_plus_one(v: dlib.sparse_vector) -> int
    
    ensures    
        - returns the dimensionality of the given sparse vector.  That is, returns a    
          number one larger than the maximum index value in the vector.  If the vector    
          is empty then returns 0.
    """
    return 0

def max_point(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    max_point(*args, **kwargs)
    Overloaded function.
    
    1. max_point(img: numpy.ndarray[(rows,cols),uint8]) -> dlib.dpoint
    
    2. max_point(img: numpy.ndarray[(rows,cols),uint16]) -> dlib.dpoint
    
    3. max_point(img: numpy.ndarray[(rows,cols),uint32]) -> dlib.dpoint
    
    4. max_point(img: numpy.ndarray[(rows,cols),uint64]) -> dlib.dpoint
    
    5. max_point(img: numpy.ndarray[(rows,cols),int8]) -> dlib.dpoint
    
    6. max_point(img: numpy.ndarray[(rows,cols),int16]) -> dlib.dpoint
    
    7. max_point(img: numpy.ndarray[(rows,cols),int32]) -> dlib.dpoint
    
    8. max_point(img: numpy.ndarray[(rows,cols),int64]) -> dlib.dpoint
    
    9. max_point(img: numpy.ndarray[(rows,cols),float32]) -> dlib.dpoint
    
    10. max_point(img: numpy.ndarray[(rows,cols),float64]) -> dlib.dpoint
    
    requires 
        - m.size > 0 
    ensures 
        - returns the location of the maximum element of the array, that is, if the 
          returned point is P then it will be the case that: img[P.y,P.x] == img.max().
    """
    pass

def max_point_interpolated(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    max_point_interpolated(*args, **kwargs)
    Overloaded function.
    
    1. max_point_interpolated(img: numpy.ndarray[(rows,cols),uint8]) -> dlib.dpoint
    
    2. max_point_interpolated(img: numpy.ndarray[(rows,cols),uint16]) -> dlib.dpoint
    
    3. max_point_interpolated(img: numpy.ndarray[(rows,cols),uint32]) -> dlib.dpoint
    
    4. max_point_interpolated(img: numpy.ndarray[(rows,cols),uint64]) -> dlib.dpoint
    
    5. max_point_interpolated(img: numpy.ndarray[(rows,cols),int8]) -> dlib.dpoint
    
    6. max_point_interpolated(img: numpy.ndarray[(rows,cols),int16]) -> dlib.dpoint
    
    7. max_point_interpolated(img: numpy.ndarray[(rows,cols),int32]) -> dlib.dpoint
    
    8. max_point_interpolated(img: numpy.ndarray[(rows,cols),int64]) -> dlib.dpoint
    
    9. max_point_interpolated(img: numpy.ndarray[(rows,cols),float32]) -> dlib.dpoint
    
    10. max_point_interpolated(img: numpy.ndarray[(rows,cols),float64]) -> dlib.dpoint
    
    requires 
        - m.size > 0 
    ensures 
        - Like max_point(), this function finds the location in m with the largest 
          value.  However, we additionally use some quadratic interpolation to find the 
          location of the maximum point with sub-pixel accuracy.  Therefore, the 
          returned point is equal to max_point(m) + some small sub-pixel delta.
    """
    pass

def min_barrier_distance(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    min_barrier_distance(*args, **kwargs)
    Overloaded function.
    
    1. min_barrier_distance(img: numpy.ndarray[(rows,cols),uint8], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),uint8]
    
    2. min_barrier_distance(img: numpy.ndarray[(rows,cols),uint16], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),uint16]
    
    3. min_barrier_distance(img: numpy.ndarray[(rows,cols),uint32], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),uint32]
    
    4. min_barrier_distance(img: numpy.ndarray[(rows,cols),uint64], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),uint64]
    
    5. min_barrier_distance(img: numpy.ndarray[(rows,cols),int8], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),int8]
    
    6. min_barrier_distance(img: numpy.ndarray[(rows,cols),int16], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),int16]
    
    7. min_barrier_distance(img: numpy.ndarray[(rows,cols),int32], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),int32]
    
    8. min_barrier_distance(img: numpy.ndarray[(rows,cols),int64], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),int64]
    
    9. min_barrier_distance(img: numpy.ndarray[(rows,cols),float32], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),float32]
    
    10. min_barrier_distance(img: numpy.ndarray[(rows,cols),float64], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),float64]
    
    11. min_barrier_distance(img: numpy.ndarray[(rows,cols,3),uint8], iterations: int=10, do_left_right_scans: bool=True) -> numpy.ndarray[(rows,cols),uint8]
    
    requires 
        - iterations > 0 
    ensures 
        - This function implements the salient object detection method described in the paper: 
            "Minimum barrier salient object detection at 80 fps" by Zhang, Jianming, et al.  
          In particular, we compute the minimum barrier distance between the borders of 
          the image and all the other pixels.  The resulting image is returned.  Note that 
          the paper talks about a bunch of other things you could do beyond computing 
          the minimum barrier distance, but this function doesn't do any of that. It's 
          just the vanilla MBD. 
        - We will perform iterations iterations of MBD passes over the image.  Larger 
          values might give better results but run slower. 
        - During each MBD iteration we make raster scans over the image.  These pass 
          from top->bottom, bottom->top, left->right, and right->left.  If 
          do_left_right_scans==false then the left/right passes are not executed. 
          Skipping them makes the algorithm about 2x faster but might reduce the 
          quality of the output.
    """
    pass

def normalize_image_gradients(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    normalize_image_gradients(*args, **kwargs)
    Overloaded function.
    
    1. normalize_image_gradients(img1: numpy.ndarray[(rows,cols),float64], img2: numpy.ndarray[(rows,cols),float64]) -> None
    
    2. normalize_image_gradients(img1: numpy.ndarray[(rows,cols),float32], img2: numpy.ndarray[(rows,cols),float32]) -> None
    
    requires 
        - img1 and img2 have the same dimensions. 
    ensures 
        - This function assumes img1 and img2 are the two gradient images produced by a 
          function like sobel_edge_detector().  It then unit normalizes the gradient 
          vectors. That is, for all valid r and c, this function ensures that: 
            - img1[r][c]*img1[r][c] + img2[r][c]*img2[r][c] == 1  
              unless both img1[r][c] and img2[r][c] were 0 initially, then they stay zero.
    """
    pass

def num_separable_filters(detector): # real signature unknown; restored from __doc__
    """
    num_separable_filters(detector: dlib.simple_object_detector) -> int
    
    Returns the number of separable filters necessary to represent the HOG filters in the given detector.
    """
    return 0

def partition_pixels(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    partition_pixels(*args, **kwargs)
    Overloaded function.
    
    1. partition_pixels(img: numpy.ndarray[(rows,cols,3),uint8]) -> int
    
    2. partition_pixels(img: numpy.ndarray[(rows,cols),uint8]) -> int
    
    3. partition_pixels(img: numpy.ndarray[(rows,cols),uint16]) -> int
    
    4. partition_pixels(img: numpy.ndarray[(rows,cols),uint32]) -> int
    
    5. partition_pixels(img: numpy.ndarray[(rows,cols),float32]) -> float
    
    6. partition_pixels(img: numpy.ndarray[(rows,cols),float64]) -> float
    
    Finds a threshold value that would be reasonable to use with 
    threshold_image(img, threshold).  It does this by finding the threshold that 
    partitions the pixels in img into two groups such that the sum of absolute 
    deviations between each pixel and the mean of its group is minimized.
    
    7. partition_pixels(img: numpy.ndarray[(rows,cols,3),uint8], num_thresholds: int) -> tuple
    
    8. partition_pixels(img: numpy.ndarray[(rows,cols),uint8], num_thresholds: int) -> tuple
    
    9. partition_pixels(img: numpy.ndarray[(rows,cols),uint16], num_thresholds: int) -> tuple
    
    10. partition_pixels(img: numpy.ndarray[(rows,cols),uint32], num_thresholds: int) -> tuple
    
    11. partition_pixels(img: numpy.ndarray[(rows,cols),float32], num_thresholds: int) -> tuple
    
    12. partition_pixels(img: numpy.ndarray[(rows,cols),float64], num_thresholds: int) -> tuple
    
    This version of partition_pixels() finds multiple partitions rather than just 
    one partition.  It does this by first partitioning the pixels just as the 
    above partition_pixels(img) does.  Then it forms a new image with only pixels 
    >= that first partition value and recursively partitions this new image. 
    However, the recursion is implemented in an efficient way which is faster than 
    explicitly forming these images and calling partition_pixels(), but the 
    output is the same as if you did.  For example, suppose you called 
    [t1,t2,t2] = partition_pixels(img,3).  Then we would have: 
       - t1 == partition_pixels(img) 
       - t2 == partition_pixels(an image with only pixels with values >= t1 in it) 
       - t3 == partition_pixels(an image with only pixels with values >= t2 in it)
    """
    pass

def polygon_area(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    polygon_area(*args, **kwargs)
    Overloaded function.
    
    1. polygon_area(pts: dlib.dpoints) -> float
    
    2. polygon_area(pts: list) -> float
    
    ensures 
        - If you walk the points pts in order to make a closed polygon, what is its 
          area?  This function returns that area.  It uses the shoelace formula to 
          compute the result and so works for general non-self-intersecting polygons.
    """
    pass

def probability_that_sequence_is_increasing(time_series): # real signature unknown; restored from __doc__
    """
    probability_that_sequence_is_increasing(time_series: object) -> float
    
    returns the probability that the given sequence of real numbers is increasing in value over time.
    """
    return 0.0

def randomly_color_image(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    randomly_color_image(*args, **kwargs)
    Overloaded function.
    
    1. randomly_color_image(img: numpy.ndarray[(rows,cols),uint8]) -> numpy.ndarray[(rows,cols,3),uint8]
    
    2. randomly_color_image(img: numpy.ndarray[(rows,cols),uint16]) -> numpy.ndarray[(rows,cols,3),uint8]
    
    3. randomly_color_image(img: numpy.ndarray[(rows,cols),uint32]) -> numpy.ndarray[(rows,cols,3),uint8]
    
    - randomly generates a mapping from gray level pixel values 
      to the RGB pixel space and then uses this mapping to create 
      a colored version of img.  Returns an image which represents 
      this colored version of img. 
    - black pixels in img will remain black in the output image.
    """
    pass

def reduce(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    reduce(*args, **kwargs)
    Overloaded function.
    
    1. reduce(df: dlib._normalized_decision_function_radial_basis, x: dlib.vectors, num_basis_vectors: int, eps: float=0.001) -> dlib._normalized_decision_function_radial_basis
    
    2. reduce(df: dlib._normalized_decision_function_radial_basis, x: numpy.ndarray[(rows,cols),float64], num_basis_vectors: int, eps: float=0.001) -> dlib._normalized_decision_function_radial_basis
    
    requires 
        - eps > 0 
        - num_bv > 0 
    ensures 
        - This routine takes a learned radial basis function and tries to find a 
          new RBF function with num_basis_vectors basis vectors that approximates 
          the given df() as closely as possible.  In particular, it finds a 
          function new_df() such that new_df(x[i])==df(x[i]) as often as possible. 
        - This is accomplished using a reduced set method that begins by using a 
          projection, in kernel space, onto a random set of num_basis_vectors 
          vectors in x.  Then, L-BFGS is used to further optimize new_df() to match 
          df().  The eps parameter controls how long L-BFGS will run, smaller 
          values of eps possibly giving better solutions but taking longer to 
          execute.
    """
    pass

def remove_incoherent_edge_pixels(line, horz_gradient, float32=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    remove_incoherent_edge_pixels(line: dlib.points, horz_gradient: numpy.ndarray[(rows,cols),float32], vert_gradient: numpy.ndarray[(rows,cols),float32], angle_thresh: float) -> dlib.points
    
    requires 
        - horz_gradient and vert_gradient have the same dimensions. 
        - horz_gradient and vert_gradient represent unit normalized vectors.  That is, 
          you should have called normalize_image_gradients(horz_gradient,vert_gradient) 
          or otherwise caused all the gradients to have unit norm. 
        - for all valid i: 
            get_rect(horz_gradient).contains(line[i]) 
    ensures 
        - This routine looks at all the points in the given line and discards the ones that 
          have outlying gradient directions.  To be specific, this routine returns a set 
          of points PTS such that:  
            - for all valid i,j: 
                - The difference in angle between the gradients for PTS[i] and PTS[j] is  
                  less than angle_threshold degrees.   
            - len(PTS) <= len(line) 
            - PTS is just line with some elements removed.
    """
    pass

def resize_image(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    resize_image(*args, **kwargs)
    Overloaded function.
    
    1. resize_image(img: numpy.ndarray[(rows,cols),uint8], rows: int, cols: int) -> numpy.ndarray[(rows,cols),uint8]
    
    2. resize_image(img: numpy.ndarray[(rows,cols),uint16], rows: int, cols: int) -> numpy.ndarray[(rows,cols),uint16]
    
    3. resize_image(img: numpy.ndarray[(rows,cols),uint32], rows: int, cols: int) -> numpy.ndarray[(rows,cols),uint32]
    
    4. resize_image(img: numpy.ndarray[(rows,cols),uint64], rows: int, cols: int) -> numpy.ndarray[(rows,cols),uint64]
    
    5. resize_image(img: numpy.ndarray[(rows,cols),int8], rows: int, cols: int) -> numpy.ndarray[(rows,cols),int8]
    
    6. resize_image(img: numpy.ndarray[(rows,cols),int16], rows: int, cols: int) -> numpy.ndarray[(rows,cols),int16]
    
    7. resize_image(img: numpy.ndarray[(rows,cols),int32], rows: int, cols: int) -> numpy.ndarray[(rows,cols),int32]
    
    8. resize_image(img: numpy.ndarray[(rows,cols),int64], rows: int, cols: int) -> numpy.ndarray[(rows,cols),int64]
    
    9. resize_image(img: numpy.ndarray[(rows,cols),float32], rows: int, cols: int) -> numpy.ndarray[(rows,cols),float32]
    
    10. resize_image(img: numpy.ndarray[(rows,cols),float64], rows: int, cols: int) -> numpy.ndarray[(rows,cols),float64]
    
    Resizes img, using bilinear interpolation, to have the indicated number of rows and columns.
    
    11. resize_image(img: numpy.ndarray[(rows,cols,3),uint8], rows: int, cols: int) -> numpy.ndarray[(rows,cols,3),uint8]
    
    Resizes img, using bilinear interpolation, to have the indicated number of rows and columns.
    
    12. resize_image(img: numpy.ndarray[(rows,cols),int8], scale: float) -> numpy.ndarray[(rows,cols),int8]
    
    13. resize_image(img: numpy.ndarray[(rows,cols),int16], scale: float) -> numpy.ndarray[(rows,cols),int16]
    
    14. resize_image(img: numpy.ndarray[(rows,cols),int32], scale: float) -> numpy.ndarray[(rows,cols),int32]
    
    15. resize_image(img: numpy.ndarray[(rows,cols),int64], scale: float) -> numpy.ndarray[(rows,cols),int64]
    
    16. resize_image(img: numpy.ndarray[(rows,cols),float32], scale: float) -> numpy.ndarray[(rows,cols),float32]
    
    17. resize_image(img: numpy.ndarray[(rows,cols),float64], scale: float) -> numpy.ndarray[(rows,cols),float64]
    
    18. resize_image(img: numpy.ndarray[(rows,cols,3),uint8], scale: float) -> numpy.ndarray[(rows,cols,3),uint8]
    
    Resizes img, using bilinear interpolation, to have the new size (img rows * scale, img cols * scale)
    """
    pass

def reverse(l): # real signature unknown; restored from __doc__
    """
    reverse(l: dlib.line) -> dlib.line
    
    ensures 
        - returns line(l.p2, l.p1) 
          (i.e. returns a line object that represents the same line as l but with the 
          endpoints, and therefore, the normal vector flipped.  This means that the 
          signed distance of operator() is also flipped).
    """
    pass

def save_face_chip(img, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    save_face_chip(img: numpy.ndarray[(rows,cols,3),uint8], face: dlib.full_object_detection, chip_filename: str, size: int=150, padding: float=0.25) -> None
    
    Takes an image and a full_object_detection that references a face in that image and saves the face with the specified file name prefix.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
    pass

def save_face_chips(img, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    save_face_chips(img: numpy.ndarray[(rows,cols,3),uint8], faces: dlib.full_object_detections, chip_filename: str, size: int=150, padding: float=0.25) -> None
    
    Takes an image and a full_object_detections object that reference faces in that image and saves the faces with the specified file name prefix.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
    pass

def save_image(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    save_image(*args, **kwargs)
    Overloaded function.
    
    1. save_image(img: numpy.ndarray[(rows,cols,3),uint8], filename: str) -> None
    
    Saves the given image to the specified path. Determines the file type from the file extension specified in the path
    
    2. save_image(img: numpy.ndarray[(rows,cols),uint8], filename: str) -> None
    
    Saves the given image to the specified path. Determines the file type from the file extension specified in the path
    """
    pass

def save_libsvm_formatted_data(file_name, samples, labels): # real signature unknown; restored from __doc__
    """
    save_libsvm_formatted_data(file_name: str, samples: dlib.sparse_vectors, labels: dlib.array) -> None
    
    requires    
        - len(samples) == len(labels)    
    ensures    
        - saves the data to the given file in libsvm format
    """
    pass

def scale_rect(rect, scale): # real signature unknown; restored from __doc__
    """
    scale_rect(rect: dlib.rectangle, scale: float) -> dlib.rectangle
    
    - return scale_rect(rect, scale) 
    (i.e. resizes the given rectangle by a scale factor)
    """
    pass

def set_dnn_prefer_smallest_algorithms(): # real signature unknown; restored from __doc__
    """
    set_dnn_prefer_smallest_algorithms() -> None
    
    Tells cuDNN to use slower algorithms that use less RAM.
    """
    pass

def shrink_rect(rect, num): # real signature unknown; restored from __doc__
    """
    shrink_rect(rect: dlib.rectangle, num: int) -> dlib.rectangle
    
     returns rectangle(rect.left()+num, rect.top()+num, rect.right()-num, rect.bottom()-num) 
      (i.e. shrinks the given rectangle by shrinking its border by num)
    """
    pass

def signed_distance_to_line(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    signed_distance_to_line(*args, **kwargs)
    Overloaded function.
    
    1. signed_distance_to_line(l: dlib.line, p: dlib.point) -> float
    
    2. signed_distance_to_line(l: dlib.line, p: dlib.dpoint) -> float
    
    ensures 
        - returns how far p is from the line l.  This is a signed distance.  The sign 
          indicates which side of the line the point is on and the magnitude is the 
          distance. Moreover, the direction of positive sign is pointed to by the 
          vector l.normal. 
        - To be specific, this routine returns dot(p-l.p1, l.normal)
    """
    pass

def skeleton(img, uint8=None): # real signature unknown; restored from __doc__
    """
    skeleton(img: numpy.ndarray[(rows,cols),uint8]) -> numpy.ndarray[(rows,cols),uint8]
    
    requires 
        - all pixels in img are set to either 255 or 0. 
    ensures 
        - This function computes the skeletonization of img and stores the result in 
          #img.  That is, given a binary image, we progressively thin the binary blobs 
          (composed of on_pixel values) until only a single pixel wide skeleton of the 
          original blobs remains. 
        - Doesn't change the shape or size of img.
    """
    pass

def sobel_edge_detector(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    sobel_edge_detector(*args, **kwargs)
    Overloaded function.
    
    1. sobel_edge_detector(img: numpy.ndarray[(rows,cols),uint8]) -> tuple
    
    2. sobel_edge_detector(img: numpy.ndarray[(rows,cols),uint16]) -> tuple
    
    3. sobel_edge_detector(img: numpy.ndarray[(rows,cols),uint32]) -> tuple
    
    4. sobel_edge_detector(img: numpy.ndarray[(rows,cols),uint64]) -> tuple
    
    5. sobel_edge_detector(img: numpy.ndarray[(rows,cols),int8]) -> tuple
    
    6. sobel_edge_detector(img: numpy.ndarray[(rows,cols),int16]) -> tuple
    
    7. sobel_edge_detector(img: numpy.ndarray[(rows,cols),int32]) -> tuple
    
    8. sobel_edge_detector(img: numpy.ndarray[(rows,cols),int64]) -> tuple
    
    9. sobel_edge_detector(img: numpy.ndarray[(rows,cols),float32]) -> tuple
    
    10. sobel_edge_detector(img: numpy.ndarray[(rows,cols),float64]) -> tuple
    
    Applies the sobel edge detector to the given input image and returns two gradient 
    images in a tuple.  The first contains the x gradients and the second contains the 
    y gradients of the image.
    """
    pass

def solve_structural_svm_problem(problem): # real signature unknown; restored from __doc__
    """
    solve_structural_svm_problem(problem: object) -> dlib.vector
    
    This function solves a structural SVM problem and returns the weight vector    
    that defines the solution.  See the example program python_examples/svm_struct.py    
    for documentation about how to create a proper problem object.
    """
    pass

def spatially_filter_image(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    spatially_filter_image(*args, **kwargs)
    Overloaded function.
    
    1. spatially_filter_image(img: numpy.ndarray[(rows,cols),uint8], filter: numpy.ndarray[(rows,cols),uint8]) -> tuple
    
    2. spatially_filter_image(img: numpy.ndarray[(rows,cols),float32], filter: numpy.ndarray[(rows,cols),float32]) -> tuple
    
    3. spatially_filter_image(img: numpy.ndarray[(rows,cols),float64], filter: numpy.ndarray[(rows,cols),float64]) -> tuple
    
    requires 
        - filter.size != 0 
    ensures 
        - Applies the given spatial filter to img and returns the result (i.e. we  
          cross-correlate img with filter).  We also return a rectangle which 
          indicates what pixels in the returned image are considered non-border pixels 
          and therefore contain output from the filter.  E.g. 
            - filtered_img,rect = spatially_filter_image(img, filter) 
          would give you the filtered image and the rectangle in question.  Since the 
          returned image has the same shape as img we fill the border pixels by setting 
          them to 0. 
     
        - The filter is applied such that it's centered over the pixel it writes its 
          output into.  For centering purposes, we consider the center element of the 
          filter to be filter[filter.shape[0]/2,filter.shape[1]/2].  This means that 
          the filter that writes its output to a pixel at location point(c,r) and is W 
          by H (width by height) pixels in size operates on exactly the pixels in the 
          rectangle centered_rect(point(c,r),W,H) within img.
    """
    pass

def spatially_filter_image_separable(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    spatially_filter_image_separable(*args, **kwargs)
    Overloaded function.
    
    1. spatially_filter_image_separable(img: numpy.ndarray[(rows,cols),uint8], row_filter: numpy.ndarray[uint8], col_filter: numpy.ndarray[uint8]) -> tuple
    
    2. spatially_filter_image_separable(img: numpy.ndarray[(rows,cols),float32], row_filter: numpy.ndarray[float32], col_filter: numpy.ndarray[float32]) -> tuple
    
    3. spatially_filter_image_separable(img: numpy.ndarray[(rows,cols),float64], row_filter: numpy.ndarray[float64], col_filter: numpy.ndarray[float64]) -> tuple
    
    requires 
        - row_filter.size != 0 
        - col_filter.size != 0 
        - row_filter and col_filter are both either row or column vectors.  
    ensures 
        - Applies the given separable spatial filter to img and returns the result 
          (i.e. we cross-correlate img with the filters).  In particular, calling this 
          function has the same effect as calling the regular spatially_filter_image() 
          routine with a filter, FILT, defined as follows:  
            - FILT(r,c) == col_filter(r)*row_filter(c) 
          Therefore, the return value of this routine is the same as if it were 
          implemented as:    
            return spatially_filter_image(img, FILT) 
          Except that this version should be faster for separable filters.
    """
    pass

def sub_image(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    sub_image(*args, **kwargs)
    Overloaded function.
    
    1. sub_image(img: array, rect: dlib.rectangle) -> array
    
    Returns a new numpy array that references the sub window in img defined by rect. 
    If rect is larger than img then rect is cropped so that it does not go outside img. 
    Therefore, this routine is equivalent to performing: 
        win = get_rect(img).intersect(rect) 
        subimg = img[win.top():win.bottom()-1,win.left():win.right()-1]
    
    2. sub_image(image_and_rect_tuple: tuple) -> array
    
    Performs: return sub_image(image_and_rect_tuple[0], image_and_rect_tuple[1])
    """
    pass

def suppress_non_maximum_edges(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    suppress_non_maximum_edges(*args, **kwargs)
    Overloaded function.
    
    1. suppress_non_maximum_edges(horz: numpy.ndarray[(rows,cols),float32], vert: numpy.ndarray[(rows,cols),float32]) -> numpy.ndarray[(rows,cols),float32]
    
    requires 
        - The two input images have the same dimensions. 
    ensures 
        - Returns an image, of the same dimensions as the input.  Each element in this 
          image holds the edge strength at that location.  Moreover, edge pixels that are not  
          local maximizers have been set to 0. 
        - let edge_strength(r,c) == sqrt(pow(horz[r][c],2) + pow(vert[r][c],2)) 
          (i.e. The Euclidean norm of the gradient) 
        - let OUT denote the returned image. 
        - for all valid r and c: 
            - if (edge_strength(r,c) is at a maximum with respect to its 2 neighboring 
              pixels along the line indicated by the image gradient vector (horz[r][c],vert[r][c])) then 
                - OUT[r][c] == edge_strength(r,c) 
            - else 
                - OUT[r][c] == 0
    
    2. suppress_non_maximum_edges(horz_and_vert_gradients: tuple) -> numpy.ndarray[(rows,cols),float32]
    
    Performs: return suppress_non_maximum_edges(horz_and_vert_gradients[0], horz_and_vert_gradients[1])
    """
    pass

def test_binary_decision_function(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    test_binary_decision_function(*args, **kwargs)
    Overloaded function.
    
    1. test_binary_decision_function(function: dlib._normalized_decision_function_radial_basis, samples: dlib.vectors, labels: dlib.array) -> binary_test
    
    2. test_binary_decision_function(function: dlib._normalized_decision_function_radial_basis, samples: numpy.ndarray[(rows,cols),float64], labels: numpy.ndarray[float64]) -> binary_test
    
    3. test_binary_decision_function(function: dlib._decision_function_linear, samples: dlib.vectors, labels: dlib.array) -> binary_test
    
    4. test_binary_decision_function(function: dlib._decision_function_sparse_linear, samples: dlib.sparse_vectors, labels: dlib.array) -> binary_test
    
    5. test_binary_decision_function(function: dlib._decision_function_radial_basis, samples: dlib.vectors, labels: dlib.array) -> binary_test
    
    6. test_binary_decision_function(function: dlib._decision_function_sparse_radial_basis, samples: dlib.sparse_vectors, labels: dlib.array) -> binary_test
    
    7. test_binary_decision_function(function: dlib._decision_function_polynomial, samples: dlib.vectors, labels: dlib.array) -> binary_test
    
    8. test_binary_decision_function(function: dlib._decision_function_sparse_polynomial, samples: dlib.sparse_vectors, labels: dlib.array) -> binary_test
    
    9. test_binary_decision_function(function: dlib._decision_function_histogram_intersection, samples: dlib.vectors, labels: dlib.array) -> binary_test
    
    10. test_binary_decision_function(function: dlib._decision_function_sparse_histogram_intersection, samples: dlib.sparse_vectors, labels: dlib.array) -> binary_test
    
    11. test_binary_decision_function(function: dlib._decision_function_sigmoid, samples: dlib.vectors, labels: dlib.array) -> binary_test
    
    12. test_binary_decision_function(function: dlib._decision_function_sparse_sigmoid, samples: dlib.sparse_vectors, labels: dlib.array) -> binary_test
    """
    pass

def test_ranking_function(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    test_ranking_function(*args, **kwargs)
    Overloaded function.
    
    1. test_ranking_function(function: dlib._decision_function_linear, samples: dlib.ranking_pairs) -> ranking_test
    
    2. test_ranking_function(function: dlib._decision_function_sparse_linear, samples: dlib.sparse_ranking_pairs) -> ranking_test
    
    3. test_ranking_function(function: dlib._decision_function_linear, sample: dlib.ranking_pair) -> ranking_test
    
    4. test_ranking_function(function: dlib._decision_function_sparse_linear, sample: dlib.sparse_ranking_pair) -> ranking_test
    """
    pass

def test_regression_function(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    test_regression_function(*args, **kwargs)
    Overloaded function.
    
    1. test_regression_function(function: dlib._decision_function_linear, samples: dlib.vectors, targets: dlib.array) -> regression_test
    
    2. test_regression_function(function: dlib._decision_function_sparse_linear, samples: dlib.sparse_vectors, targets: dlib.array) -> regression_test
    
    3. test_regression_function(function: dlib._decision_function_radial_basis, samples: dlib.vectors, targets: dlib.array) -> regression_test
    
    4. test_regression_function(function: dlib._decision_function_sparse_radial_basis, samples: dlib.sparse_vectors, targets: dlib.array) -> regression_test
    
    5. test_regression_function(function: dlib._decision_function_histogram_intersection, samples: dlib.vectors, targets: dlib.array) -> regression_test
    
    6. test_regression_function(function: dlib._decision_function_sparse_histogram_intersection, samples: dlib.sparse_vectors, targets: dlib.array) -> regression_test
    
    7. test_regression_function(function: dlib._decision_function_sigmoid, samples: dlib.vectors, targets: dlib.array) -> regression_test
    
    8. test_regression_function(function: dlib._decision_function_sparse_sigmoid, samples: dlib.sparse_vectors, targets: dlib.array) -> regression_test
    
    9. test_regression_function(function: dlib._decision_function_polynomial, samples: dlib.vectors, targets: dlib.array) -> regression_test
    
    10. test_regression_function(function: dlib._decision_function_sparse_polynomial, samples: dlib.sparse_vectors, targets: dlib.array) -> regression_test
    """
    pass

def test_sequence_segmenter(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    test_sequence_segmenter(*args, **kwargs)
    Overloaded function.
    
    1. test_sequence_segmenter(arg0: dlib.segmenter_type, arg1: dlib.vectorss, arg2: dlib.rangess) -> dlib.segmenter_test
    
    2. test_sequence_segmenter(arg0: dlib.segmenter_type, arg1: dlib.sparse_vectorss, arg2: dlib.rangess) -> dlib.segmenter_test
    """
    pass

def test_shape_predictor(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    test_shape_predictor(*args, **kwargs)
    Overloaded function.
    
    1. test_shape_predictor(dataset_filename: str, predictor_filename: str) -> float
    
    ensures 
        - Loads an image dataset from dataset_filename.  We assume dataset_filename is 
          a file using the XML format written by save_image_dataset_metadata(). 
        - Loads a shape_predictor from the file predictor_filename.  This means 
          predictor_filename should be a file produced by the train_shape_predictor() 
          routine. 
        - This function tests the predictor against the dataset and returns the 
          mean average error of the detector.  In fact, The 
          return value of this function is identical to that of dlib's 
          shape_predictor_trainer() routine.  Therefore, see the documentation 
          for shape_predictor_trainer() for a detailed definition of the mean average error.
    
    2. test_shape_predictor(images: list, detections: list, shape_predictor: dlib.shape_predictor) -> float
    
    requires 
        - len(images) == len(object_detections) 
        - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
        - object_detections should be a list of lists of dlib.full_object_detection objects.       Each dlib.full_object_detection contains the bounding box and the lists of points that make up the object parts.
     ensures 
        - shape_predictor should be a file produced by the train_shape_predictor()  
          routine. 
        - This function tests the predictor against the dataset and returns the 
          mean average error of the detector.  In fact, The 
          return value of this function is identical to that of dlib's 
          shape_predictor_trainer() routine.  Therefore, see the documentation 
          for shape_predictor_trainer() for a detailed definition of the mean average error.
    
    3. test_shape_predictor(images: list, detections: list, scales: list, shape_predictor: dlib.shape_predictor) -> float
    
    requires 
        - len(images) == len(object_detections) 
        - len(object_detections) == len(scales) 
        - for every sublist in object_detections: len(object_detections[i]) == len(scales[i]) 
        - scales is a list of floating point scales that each predicted part location       should be divided by. Useful for normalization. 
        - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
        - object_detections should be a list of lists of dlib.full_object_detection objects.       Each dlib.full_object_detection contains the bounding box and the lists of points that make up the object parts.
     ensures 
        - shape_predictor should be a file produced by the train_shape_predictor()  
          routine. 
        - This function tests the predictor against the dataset and returns the 
          mean average error of the detector.  In fact, The 
          return value of this function is identical to that of dlib's 
          shape_predictor_trainer() routine.  Therefore, see the documentation 
          for shape_predictor_trainer() for a detailed definition of the mean average error.
    """
    pass

def test_simple_object_detector(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    test_simple_object_detector(*args, **kwargs)
    Overloaded function.
    
    1. test_simple_object_detector(dataset_filename: str, detector_filename: str, upsampling_amount: int=-1) -> dlib.simple_test_results
    
    ensures 
                    - Loads an image dataset from dataset_filename.  We assume dataset_filename is 
                      a file using the XML format written by save_image_dataset_metadata(). 
                    - Loads a simple_object_detector from the file detector_filename.  This means 
                      detector_filename should be a file produced by the train_simple_object_detector()  
                      routine. 
                    - This function tests the detector against the dataset and returns the 
                      precision, recall, and average precision of the detector.  In fact, The 
                      return value of this function is identical to that of dlib's 
                      test_object_detection_function() routine.  Therefore, see the documentation 
                      for test_object_detection_function() for a detailed definition of these 
                      metrics. 
                    - if upsampling_amount>=0 then we upsample the data by upsampling_amount rather than 
                      use any upsampling amount that happens to be encoded in the given detector.  If upsampling_amount<0 
                      then we use the upsampling amount the detector wants to use.
    
    2. test_simple_object_detector(dataset_filename: str, detector: dlib::simple_object_detector_py, upsampling_amount: int=-1) -> dlib.simple_test_results
    
    ensures 
                    - Loads an image dataset from dataset_filename.  We assume dataset_filename is 
                      a file using the XML format written by save_image_dataset_metadata(). 
                    - Loads a simple_object_detector from the file detector_filename.  This means 
                      detector_filename should be a file produced by the train_simple_object_detector()  
                      routine. 
                    - This function tests the detector against the dataset and returns the 
                      precision, recall, and average precision of the detector.  In fact, The 
                      return value of this function is identical to that of dlib's 
                      test_object_detection_function() routine.  Therefore, see the documentation 
                      for test_object_detection_function() for a detailed definition of these 
                      metrics. 
                    - if upsampling_amount>=0 then we upsample the data by upsampling_amount rather than 
                      use any upsampling amount that happens to be encoded in the given detector.  If upsampling_amount<0 
                      then we use the upsampling amount the detector wants to use.
    
    3. test_simple_object_detector(images: list, boxes: list, detector: dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6u>, dlib::default_fhog_feature_extractor> >, upsampling_amount: int=0) -> dlib.simple_test_results
    
    requires 
                   - len(images) == len(boxes) 
                   - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
                   - boxes should be a list of lists of dlib.rectangle object. 
                   - Optionally, take the number of times to upsample the testing images (upsampling_amount >= 0). 
                 ensures 
                   - Loads a simple_object_detector from the file detector_filename.  This means 
                     detector_filename should be a file produced by the train_simple_object_detector() 
                     routine. 
                   - This function tests the detector against the dataset and returns the 
                     precision, recall, and average precision of the detector.  In fact, The 
                     return value of this function is identical to that of dlib's 
                     test_object_detection_function() routine.  Therefore, see the documentation 
                     for test_object_detection_function() for a detailed definition of these 
                     metrics. 
    
    4. test_simple_object_detector(images: list, boxes: list, detector: dlib::simple_object_detector_py, upsampling_amount: int=-1) -> dlib.simple_test_results
    
    requires 
                   - len(images) == len(boxes) 
                   - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
                   - boxes should be a list of lists of dlib.rectangle object. 
                 ensures 
                   - Loads a simple_object_detector from the file detector_filename.  This means 
                     detector_filename should be a file produced by the train_simple_object_detector() 
                     routine. 
                   - This function tests the detector against the dataset and returns the 
                     precision, recall, and average precision of the detector.  In fact, The 
                     return value of this function is identical to that of dlib's 
                     test_object_detection_function() routine.  Therefore, see the documentation 
                     for test_object_detection_function() for a detailed definition of these 
                     metrics.
    """
    pass

def threshold_filter_singular_values(detector, thresh): # real signature unknown; restored from __doc__
    """
    threshold_filter_singular_values(detector: dlib.simple_object_detector, thresh: float) -> dlib.simple_object_detector
    
    requires 
        - thresh >= 0 
    ensures 
        - Removes all components of the filters in the given detector that have 
          singular values that are smaller than the given threshold.  Therefore, this 
          function allows you to control how many separable filters are in a detector. 
          In particular, as thresh gets larger the quantity 
          num_separable_filters(threshold_filter_singular_values(detector,thresh)) 
          will generally get smaller and therefore give a faster running detector. 
          However, note that at some point a large enough thresh will drop too much 
          information from the filters and their accuracy will suffer.   
        - returns the updated detector
    """
    pass

def threshold_image(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    threshold_image(*args, **kwargs)
    Overloaded function.
    
    1. threshold_image(img: numpy.ndarray[(rows,cols),uint8]) -> numpy.ndarray[(rows,cols),uint8]
    
    2. threshold_image(img: numpy.ndarray[(rows,cols),uint16]) -> numpy.ndarray[(rows,cols),uint8]
    
    3. threshold_image(img: numpy.ndarray[(rows,cols),uint32]) -> numpy.ndarray[(rows,cols),uint8]
    
    4. threshold_image(img: numpy.ndarray[(rows,cols),float32]) -> numpy.ndarray[(rows,cols),uint8]
    
    5. threshold_image(img: numpy.ndarray[(rows,cols),float64]) -> numpy.ndarray[(rows,cols),uint8]
    
    6. threshold_image(img: numpy.ndarray[(rows,cols,3),uint8]) -> numpy.ndarray[(rows,cols),uint8]
    
    Thresholds img and returns the result.  Pixels in img with grayscale values >= partition_pixels(img) 
    have an output value of 255 and all others have a value of 0.
    
    7. threshold_image(img: numpy.ndarray[(rows,cols),uint8], thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    8. threshold_image(img: numpy.ndarray[(rows,cols),uint16], thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    9. threshold_image(img: numpy.ndarray[(rows,cols),uint32], thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    10. threshold_image(img: numpy.ndarray[(rows,cols),float32], thresh: float) -> numpy.ndarray[(rows,cols),uint8]
    
    11. threshold_image(img: numpy.ndarray[(rows,cols),float64], thresh: float) -> numpy.ndarray[(rows,cols),uint8]
    
    12. threshold_image(img: numpy.ndarray[(rows,cols,3),uint8], thresh: int) -> numpy.ndarray[(rows,cols),uint8]
    
    Thresholds img and returns the result.  Pixels in img with grayscale values >= thresh 
    have an output value of 255 and all others have a value of 0.
    """
    pass

def tile_images(images): # real signature unknown; restored from __doc__
    """
    tile_images(images: list) -> array
    
    requires 
        - images is a list of numpy arrays that can be interpreted as images.  They 
          must all be the same type of image as well. 
    ensures 
        - This function takes the given images and tiles them into a single large 
          square image and returns this new big tiled image.  Therefore, it is a 
          useful method to visualize many small images at once.
    """
    return array

def train_sequence_segmenter(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    train_sequence_segmenter(*args, **kwargs)
    Overloaded function.
    
    1. train_sequence_segmenter(samples: dlib.vectorss, segments: dlib.rangess, params: dlib.segmenter_params=<BIO,highFeats,signed,win=5,threads=4,eps=0.1,cache=40,non-verbose,C=100>) -> dlib.segmenter_type
    
    2. train_sequence_segmenter(samples: dlib.sparse_vectorss, segments: dlib.rangess, params: dlib.segmenter_params=<BIO,highFeats,signed,win=5,threads=4,eps=0.1,cache=40,non-verbose,C=100>) -> dlib.segmenter_type
    """
    pass

def train_shape_predictor(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    train_shape_predictor(*args, **kwargs)
    Overloaded function.
    
    1. train_shape_predictor(images: list, object_detections: list, options: dlib.shape_predictor_training_options) -> dlib.shape_predictor
    
    requires 
        - options.lambda_param > 0 
        - 0 < options.nu <= 1 
        - options.feature_pool_region_padding >= 0 
        - len(images) == len(object_detections) 
        - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
        - object_detections should be a list of lists of dlib.full_object_detection objects.       Each dlib.full_object_detection contains the bounding box and the lists of points that make up the object parts.
    ensures 
        - Uses dlib's shape_predictor_trainer object to train a 
          shape_predictor based on the provided labeled images, full_object_detections, and options.
        - The trained shape_predictor is returned
    
    2. train_shape_predictor(dataset_filename: str, predictor_output_filename: str, options: dlib.shape_predictor_training_options) -> None
    
    requires 
        - options.lambda_param > 0 
        - 0 < options.nu <= 1 
        - options.feature_pool_region_padding >= 0 
    ensures 
        - Uses dlib's shape_predictor_trainer to train a 
          shape_predictor based on the labeled images in the XML file 
          dataset_filename and the provided options.  This function assumes the file dataset_filename is in the 
          XML format produced by dlib's save_image_dataset_metadata() routine. 
        - The trained shape predictor is serialized to the file predictor_output_filename.
    """
    pass

def train_simple_object_detector(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    train_simple_object_detector(*args, **kwargs)
    Overloaded function.
    
    1. train_simple_object_detector(dataset_filename: str, detector_output_filename: str, options: dlib.simple_object_detector_training_options) -> None
    
    requires 
        - options.C > 0 
    ensures 
        - Uses the structural_object_detection_trainer to train a 
          simple_object_detector based on the labeled images in the XML file 
          dataset_filename.  This function assumes the file dataset_filename is in the 
          XML format produced by dlib's save_image_dataset_metadata() routine. 
        - This function will apply a reasonable set of default parameters and 
          preprocessing techniques to the training procedure for simple_object_detector 
          objects.  So the point of this function is to provide you with a very easy 
          way to train a basic object detector.   
        - The trained object detector is serialized to the file detector_output_filename.
    
    2. train_simple_object_detector(images: list, boxes: list, options: dlib.simple_object_detector_training_options) -> dlib::simple_object_detector_py
    
    requires 
        - options.C > 0 
        - len(images) == len(boxes) 
        - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
        - boxes should be a list of lists of dlib.rectangle object. 
    ensures 
        - Uses the structural_object_detection_trainer to train a 
          simple_object_detector based on the labeled images and bounding boxes.  
        - This function will apply a reasonable set of default parameters and 
          preprocessing techniques to the training procedure for simple_object_detector 
          objects.  So the point of this function is to provide you with a very easy 
          way to train a basic object detector.   
        - The trained object detector is returned.
    """
    pass

def transform_image(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    transform_image(*args, **kwargs)
    Overloaded function.
    
    1. transform_image(img: numpy.ndarray[(rows,cols),uint8], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),uint8]
    
    2. transform_image(img: numpy.ndarray[(rows,cols),uint16], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),uint16]
    
    3. transform_image(img: numpy.ndarray[(rows,cols),uint32], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),uint32]
    
    4. transform_image(img: numpy.ndarray[(rows,cols),uint64], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),uint64]
    
    5. transform_image(img: numpy.ndarray[(rows,cols),int8], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),int8]
    
    6. transform_image(img: numpy.ndarray[(rows,cols),int16], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),int16]
    
    7. transform_image(img: numpy.ndarray[(rows,cols),int32], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),int32]
    
    8. transform_image(img: numpy.ndarray[(rows,cols),int64], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),int64]
    
    9. transform_image(img: numpy.ndarray[(rows,cols),float32], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),float32]
    
    10. transform_image(img: numpy.ndarray[(rows,cols),float64], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols),float64]
    
    11. transform_image(img: numpy.ndarray[(rows,cols,3),uint8], map_point: dlib.point_transform_projective, rows: int, columns: int) -> numpy.ndarray[(rows,cols,3),uint8]
    
    requires 
        - rows > 0 
        - columns > 0 
    ensures 
        - Returns an image that is the given rows by columns in size and contains a 
          transformed part of img.  To do this, we interpret map_point as a mapping 
          from pixels in the returned image to pixels in the input img.  transform_image()  
          uses this mapping and bilinear interpolation to fill the output image with an 
          interpolated copy of img.   
        - Any locations in the output image that map to pixels outside img are set to 0.
    """
    pass

def translate_rect(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    translate_rect(*args, **kwargs)
    Overloaded function.
    
    1. translate_rect(rect: dlib.rectangle, p: dlib.point) -> dlib.rectangle
    
     returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) 
      (i.e. moves the location of the rectangle but doesn't change its shape)
    
    2. translate_rect(rect: dlib.drectangle, p: dlib.point) -> dlib.drectangle
    
     returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) 
      (i.e. moves the location of the rectangle but doesn't change its shape)
    
    3. translate_rect(rect: dlib.rectangle, p: dlib.dpoint) -> dlib.rectangle
    
     returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) 
      (i.e. moves the location of the rectangle but doesn't change its shape)
    
    4. translate_rect(rect: dlib.drectangle, p: dlib.dpoint) -> dlib.drectangle
    
     returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) 
      (i.e. moves the location of the rectangle but doesn't change its shape)
    """
    pass

def zero_border_pixels(*args, **kwargs): # real signature unknown; restored from __doc__
    """
    zero_border_pixels(*args, **kwargs)
    Overloaded function.
    
    1. zero_border_pixels(img: numpy.ndarray[(rows,cols),uint8], x_border_size: int, y_border_size: int) -> None
    
    2. zero_border_pixels(img: numpy.ndarray[(rows,cols),uint16], x_border_size: int, y_border_size: int) -> None
    
    3. zero_border_pixels(img: numpy.ndarray[(rows,cols),uint32], x_border_size: int, y_border_size: int) -> None
    
    4. zero_border_pixels(img: numpy.ndarray[(rows,cols),uint64], x_border_size: int, y_border_size: int) -> None
    
    5. zero_border_pixels(img: numpy.ndarray[(rows,cols),int8], x_border_size: int, y_border_size: int) -> None
    
    6. zero_border_pixels(img: numpy.ndarray[(rows,cols),int16], x_border_size: int, y_border_size: int) -> None
    
    7. zero_border_pixels(img: numpy.ndarray[(rows,cols),int32], x_border_size: int, y_border_size: int) -> None
    
    8. zero_border_pixels(img: numpy.ndarray[(rows,cols),int64], x_border_size: int, y_border_size: int) -> None
    
    9. zero_border_pixels(img: numpy.ndarray[(rows,cols),float32], x_border_size: int, y_border_size: int) -> None
    
    10. zero_border_pixels(img: numpy.ndarray[(rows,cols),float64], x_border_size: int, y_border_size: int) -> None
    
    11. zero_border_pixels(img: numpy.ndarray[(rows,cols,3),uint8], x_border_size: int, y_border_size: int) -> None
    
    requires 
        - x_border_size >= 0 
        - y_border_size >= 0 
    ensures 
        - The size and shape of img isn't changed by this function. 
        - for all valid r such that r+y_border_size or r-y_border_size gives an invalid row 
            - for all valid c such that c+x_border_size or c-x_border_size gives an invalid column  
                - assigns the pixel img[r][c] to 0.  
                  (i.e. assigns 0 to every pixel in the border of img)
    
    12. zero_border_pixels(img: numpy.ndarray[(rows,cols),uint8], inside: dlib.rectangle) -> None
    
    13. zero_border_pixels(img: numpy.ndarray[(rows,cols),uint16], inside: dlib.rectangle) -> None
    
    14. zero_border_pixels(img: numpy.ndarray[(rows,cols),uint32], inside: dlib.rectangle) -> None
    
    15. zero_border_pixels(img: numpy.ndarray[(rows,cols),uint64], inside: dlib.rectangle) -> None
    
    16. zero_border_pixels(img: numpy.ndarray[(rows,cols),int8], inside: dlib.rectangle) -> None
    
    17. zero_border_pixels(img: numpy.ndarray[(rows,cols),int16], inside: dlib.rectangle) -> None
    
    18. zero_border_pixels(img: numpy.ndarray[(rows,cols),int32], inside: dlib.rectangle) -> None
    
    19. zero_border_pixels(img: numpy.ndarray[(rows,cols),int64], inside: dlib.rectangle) -> None
    
    20. zero_border_pixels(img: numpy.ndarray[(rows,cols),float32], inside: dlib.rectangle) -> None
    
    21. zero_border_pixels(img: numpy.ndarray[(rows,cols),float64], inside: dlib.rectangle) -> None
    
    22. zero_border_pixels(img: numpy.ndarray[(rows,cols,3),uint8], inside: dlib.rectangle) -> None
    
    ensures 
        - The size and shape of img isn't changed by this function. 
        - All the pixels in img that are not contained inside the inside rectangle 
          given to this function are set to 0.  That is, anything not "inside" is on 
          the border and set to 0.
    """
    pass

# classes

from .array import array
from .cca_outputs import cca_outputs
from .chip_details import chip_details
from .chip_dims import chip_dims
from .cnn_face_detection_model_v1 import cnn_face_detection_model_v1
from .correlation_tracker import correlation_tracker
from .dpoint import dpoint
from .dpoints import dpoints
from .drectangle import drectangle
from .face_recognition_model_v1 import face_recognition_model_v1
from .fhog_object_detector import fhog_object_detector
from .full_object_detection import full_object_detection
from .full_object_detections import full_object_detections
from .function_evaluation import function_evaluation
from .function_evaluation_request import function_evaluation_request
from .function_spec import function_spec
from .global_function_search import global_function_search
from .hough_transform import hough_transform
from .image_gradients import image_gradients
from .line import line
from .matrix import matrix
from .mmod_rectangle import mmod_rectangle
from .mmod_rectangles import mmod_rectangles
from .mmod_rectangless import mmod_rectangless
from .no_convex_quadrilateral import no_convex_quadrilateral
from .pair import pair
from .point import point
from .points import points
from .point_transform_projective import point_transform_projective
from .pyramid_down import pyramid_down
from .range import range
from .ranges import ranges
from .rangess import rangess
from .ranking_pair import ranking_pair
from .ranking_pairs import ranking_pairs
from .rectangle import rectangle
from .rectangles import rectangles
from .rectangless import rectangless
from .rect_filter import rect_filter
from .rgb_pixel import rgb_pixel
from .rvm_trainer_histogram_intersection import rvm_trainer_histogram_intersection
from .rvm_trainer_linear import rvm_trainer_linear
from .rvm_trainer_radial_basis import rvm_trainer_radial_basis
from .rvm_trainer_sparse_histogram_intersection import rvm_trainer_sparse_histogram_intersection
from .rvm_trainer_sparse_linear import rvm_trainer_sparse_linear
from .rvm_trainer_sparse_radial_basis import rvm_trainer_sparse_radial_basis
from .segmenter_params import segmenter_params
from .segmenter_test import segmenter_test
from .segmenter_type import segmenter_type
from .shape_predictor import shape_predictor
from .shape_predictor_training_options import shape_predictor_training_options
from .simple_object_detector import simple_object_detector
from .simple_object_detector_training_options import simple_object_detector_training_options
from .simple_test_results import simple_test_results
from .sparse_ranking_pair import sparse_ranking_pair
from .sparse_ranking_pairs import sparse_ranking_pairs
from .sparse_vector import sparse_vector
from .sparse_vectors import sparse_vectors
from .sparse_vectorss import sparse_vectorss
from .svm_c_trainer_histogram_intersection import svm_c_trainer_histogram_intersection
from .svm_c_trainer_linear import svm_c_trainer_linear
from .svm_c_trainer_radial_basis import svm_c_trainer_radial_basis
from .svm_c_trainer_sparse_histogram_intersection import svm_c_trainer_sparse_histogram_intersection
from .svm_c_trainer_sparse_linear import svm_c_trainer_sparse_linear
from .svm_c_trainer_sparse_radial_basis import svm_c_trainer_sparse_radial_basis
from .svm_rank_trainer import svm_rank_trainer
from .svm_rank_trainer_sparse import svm_rank_trainer_sparse
from .vector import vector
from .vectors import vectors
from .vectorss import vectorss
from ._binary_test import _binary_test
from ._decision_function_histogram_intersection import _decision_function_histogram_intersection
from ._decision_function_linear import _decision_function_linear
from ._decision_function_polynomial import _decision_function_polynomial
from ._decision_function_radial_basis import _decision_function_radial_basis
from ._decision_function_sigmoid import _decision_function_sigmoid
from ._decision_function_sparse_histogram_intersection import _decision_function_sparse_histogram_intersection
from ._decision_function_sparse_linear import _decision_function_sparse_linear
from ._decision_function_sparse_polynomial import _decision_function_sparse_polynomial
from ._decision_function_sparse_radial_basis import _decision_function_sparse_radial_basis
from ._decision_function_sparse_sigmoid import _decision_function_sparse_sigmoid
from ._linear_kernel import _linear_kernel
from ._normalized_decision_function_radial_basis import _normalized_decision_function_radial_basis
from ._radial_basis_kernel import _radial_basis_kernel
from ._range_iter import _range_iter
from ._ranking_test import _ranking_test
from ._regression_test import _regression_test
from ._row import _row
# variables with complex values

__loader__ = None # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x10b51f5c0>'

__spec__ = None # (!) real value is "ModuleSpec(name='dlib', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x10b51f5c0>, origin='/Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so')"

