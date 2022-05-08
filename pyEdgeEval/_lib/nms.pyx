import numpy as np


cdef extern from "Matrix.hh":
    cppclass Matrix:
        Matrix () except +
        Matrix (int rows, int cols) except +
        Matrix (int rows, int cols, double* data) except +
        double* data ()

cdef extern from "benms.hh":
    void benms(Matrix& out, const Matrix& edge, const Matrix& ori,
               int r, int s, float m)


cdef _nms(double[::1,:] out, double[::1,:] edge, double[::1,:] ori, int r, int s, float m):

    cdef int rows = edge.shape[0]
    cdef int cols = edge.shape[1]

    # need to convert numpy to Matrix
    # FIXME: implement a version without dependencies for Matrix
    cdef Matrix medge = Matrix(rows, cols)
    cdef Matrix mori = Matrix(rows, cols)
    cdef double[::1,:] medge_view = <double[:edge.shape[0]:1,:edge.shape[1]]>medge.data()
    cdef double[::1,:] mori_view = <double[:ori.shape[0]:1,:ori.shape[1]]>mori.data()
    medge_view[:,:] = edge[:,:]
    mori_view[:,:] = ori[:,:]

    # define output
    cdef Matrix mout

    benms(mout, medge, mori, r, s, m)

    cdef double[::1,:] mout_view = <double[:out.shape[0]:1,:out.shape[1]]>mout.data()
    out[:,:] = mout_view[:,:]


def nms(edge, ori, r=1, s=5, m=1.01):
    if edge.shape != ori.shape:
        raise ValueError('edge.shape ({}) and ori.shape ({}) shape do not match'.format(edge.shape, ori.shape))

    _edge = edge.astype('float64').copy(order='F')
    _ori = ori.astype('float64').copy(order='F')
    _out = np.zeros_like(_edge, order='F')

    _r = int(r)
    _s = int(s)
    _m = float(m)

    _nms(_out, _edge, _ori, _r, _s, _m)

    return _out
