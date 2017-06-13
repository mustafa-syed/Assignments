import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """

    orig_shape = x.shape

    # additional code to check shape
    if np.count_nonzero(orig_shape) != np.size(orig_shape):
        print "shape error"
        return x

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        # METHOD - I (changing axis)
        # xMax = np.max(x, axis=1)
        # xMax = xMax[:,np.newaxis]   # converting from row to column vector
        # x = x - xMax                # subtracting maximum of each row from it
        # x = np.exp(x)
        # xSum = np.sum(x, axis=1)    # row-wise sum vector
        # xSum = xSum[:, np.newaxis]  # converting from row to column vector
        # x = x/xSum                  # broadcasting

        # METHOD - II (Transpose)
        xMax = np.max(x, axis=1)
        x = x.T - xMax  # subtracting maximum of each row from it
        x = np.exp(x)
        xSum = np.sum(x, axis=0)  # row-wise sum vector
        x = (x / xSum).T  # broadcasting
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x -= np.max(x)
        x = np.exp(x)
        return x/np.sum(x)
        #raise NotImplementedError
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    # array of shape 5x2
    x = np.array(np.arange(10))
    x = x.reshape([5,2])
    print x
    x=softmax(x)
    print x

    # array of shape 2x5
    x = np.array(np.arange(10))
    x = x.reshape([2, 5])
    print x
    x = softmax(x)
    print x

    x = np.array([])
    print x
    x=softmax(x)
    print x
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
