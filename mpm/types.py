import ctypes
from ctypes import c_void_p, c_uint64, c_int32, c_ulonglong, c_size_t, c_float, c_longlong, c_bool
import numpy as np

import os
import time

src_dir = os.path.join(os.path.dirname(__file__))

start_time = time.time()
print("LOADING....")
if os.environ.get("COMPILE_MPM") or not os.path.isfile(f"{src_dir}/libmaniskill_mpm.so"):
    command = f"nvcc -O3 -Xptxas -O3,-v -ccbin=g++ --compiler-options -fPIC -shared {src_dir}/csrc/integrator.cu -o {src_dir}/libmaniskill_mpm.so"
    print(command)
    if os.system(command) != 0:
        print("Failed to compile mpm library")
        exit()

lib = ctypes.cdll.LoadLibrary(f"{src_dir}/libmaniskill_mpm.so")
print(f"Loaded with {time.time() - start_time} secs.")

cuda_stream_t = c_void_p
texture_t = c_ulonglong


class vec3(ctypes.Array):
    _length_ = 3
    _type_ = ctypes.c_float

    length = 3
    ctype = ctypes.c_float
    size = 12
    np_type = np.float32

    def __repr__(self):
        return f"ivec3({self[0]}, {self[1]}, {self[2]})"


class quat(ctypes.Array):
    _length_ = 4
    _type_ = ctypes.c_float

    length = 4
    ctype = ctypes.c_float
    size = 16
    np_type = np.float32

    def __repr__(self):
        return f"quat({self[0]}, {self[1]}, {self[2]}, {self[3]})"


class ivec3(ctypes.Array):
    _length_ = 3
    _type_ = ctypes.c_int

    length = 3
    ctype = ctypes.c_int
    size = 12
    np_type = np.int32

    def __repr__(self):
        return f"ivec3({self[0]}, {self[1]}, {self[2]})"


class mat3(ctypes.Array):
    _length_ = 9
    _type_ = ctypes.c_float

    length = 9
    ctype = ctypes.c_float
    size = 36
    np_type = np.float32

    def __repr__(self):
        return f"ivec3({self[0]}, {self[1]}, {self[2]}, {self[3]}, {self[4]}, {self[5]}, {self[6]}, {self[7]}, {self[8]})"


class texture_resources(ctypes.Structure):
    _fields_ = [("array", c_void_p), ("texture", texture_t)]


class float32:
    length = 1
    ctype = ctypes.c_float
    size = 4
    np_type = np.float32


class int32:
    length = 1
    ctype = ctypes.c_int
    size = 4
    np_type = np.int32


class int64:
    length = 1
    ctype = ctypes.c_int64
    size = 8
    np_type = np.int64


lib.cuda_alloc.restype = c_void_p
lib.cuda_alloc.argtypes = [c_size_t]

lib.cuda_free.argtypes = [c_void_p]

lib.cuda_upload.argtypes = [c_void_p, c_void_p, c_size_t]
lib.cuda_upload_async.argtypes = [c_void_p, c_void_p, c_size_t, cuda_stream_t]

lib.cuda_download.argtypes = [c_void_p, c_void_p, c_size_t]
lib.cuda_download_async.argtypes = [c_void_p, c_void_p, c_size_t, cuda_stream_t]

lib.cuda_copy.argtypes = [c_void_p, c_void_p, c_size_t]

lib.cuda_copy2d.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t, c_size_t, c_size_t]

lib.cuda_zero.argtypes = [c_void_p, c_size_t]
lib.cuda_zero_async.argtypes = [c_void_p, c_size_t, cuda_stream_t]

lib.create_volume.argtypes = [c_void_p, c_int32, c_int32]
lib.create_volume.restype = texture_resources

lib.destroy_volume.argtypes = [texture_resources]

lib.cuda_stream_create.restype = cuda_stream_t
lib.cuda_stream_destroy.argtypes = [cuda_stream_t]
lib.cuda_stream_sync.argtypes = [cuda_stream_t]

lib.compute_grid_lower.argtypes = [
    c_void_p,
    c_float,
    c_float,
    c_void_p,
    c_int32,
    cuda_stream_t,
]

lib.compute_svd.argtypes = [
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_float,
    c_int32,
    cuda_stream_t,
]

lib.compute_svd_grad.argtypes = [
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,

    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,

    c_void_p,
    c_float,
    c_int32,
    cuda_stream_t,
]


lib.p2g.argtypes = [
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,

    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,

    c_void_p,

    ctypes.POINTER(ivec3),
    # ivec3,

    c_float,
    c_float,
    c_float,
    c_void_p,
    c_void_p,
    c_void_p,
    c_int32,
    cuda_stream_t,
]


lib.p2g_grad.argtypes = [
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,

    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,

    c_void_p,

    ctypes.POINTER(ivec3),
    # ivec3,

    c_float,
    c_float,
    c_float,
    c_void_p,
    c_void_p,
    c_void_p,

    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,

    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,

    c_int32,
    cuda_stream_t,
]

lib.grid_op_v2.argtypes = [c_void_p] * 11 + [c_float] * 4 + [c_void_p, ctypes.POINTER(ivec3), c_int32, cuda_stream_t]

lib.grid_op_v2_grad.argtypes = [c_void_p] * 11 + [c_void_p] * 6 + [c_float] * 4 + [c_void_p, c_void_p, ctypes.POINTER(ivec3), c_int32, cuda_stream_t]

lib.g2p.argtypes = [
    c_void_p,
    c_void_p,
    c_void_p,
    c_float,
    c_float,
    c_float,
    ctypes.POINTER(ivec3),
    c_void_p,
    c_float,
    c_void_p,
    c_void_p,
    c_int32,
    cuda_stream_t,
]

lib.g2p_grad.argtypes = [
    c_void_p,
    c_void_p,
    c_void_p,
    c_float,
    c_float,
    c_float,
    ctypes.POINTER(ivec3),
    c_void_p,
    c_float,
    c_void_p,
    c_void_p,
    c_int32,

    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,

    cuda_stream_t,
]

lib.render.argtypes = [c_void_p] * 13 + [c_float, ctypes.POINTER(ivec3), c_int32, c_bool, ctypes.POINTER(ivec3), c_int32, c_int32, c_float, c_int32, c_void_p, cuda_stream_t]

lib.particle_sdf.argtypes = [c_void_p] * 5 + [c_int32, ctypes.POINTER(ivec3), c_float] + [c_void_p] * 3 + [c_int32, cuda_stream_t]

lib.compute_dist.argtypes = [c_void_p] * 6 + [c_int32] + [c_void_p] * 4 + [c_int32, c_int32, cuda_stream_t]

lib.particle2mass.argtypes = [c_void_p] * 3 + [ctypes.POINTER(ivec3), c_float, c_float] + [c_void_p] * 4 + [c_int32, c_int32, c_int32, cuda_stream_t]



class array:
    def __init__(self, dtype=float32, length=0):
        assert length > 0
        if dtype == float:
            dtype = float32
        if dtype == int:
            dtype = int32
        self.dtype = dtype

        self.bytes = dtype.size
        self.nbytes = dtype.size * length
        self.data_ptr = lib.cuda_alloc(self.nbytes)

        shape = (length,)
        if dtype.length != 1:
            shape = shape + (dtype.length,)
        self.shape = shape

        self.zero()

    def upload_async(self, arr: np.ndarray, stream):
        # only for pos
        n = arr.shape[0]
        assert arr.shape[0] == self.shape[0]
        data_size = self.bytes * n

        assert self.dtype.np_type == arr.dtype
        assert self.shape == arr.shape
        assert self.nbytes == arr.nbytes
        assert arr.data.contiguous
        ptr = arr.__array_interface__["data"][0]
        lib.cuda_upload_async(self.data_ptr, ptr, data_size, stream)

    def upload(self, arr: np.ndarray, strict=False):
        arr = np.ascontiguousarray(arr, dtype=self.dtype.np_type)
        n = arr.shape[0]
        if strict:
            assert n == self.shape[0]
        assert n <= self.shape[0], f"{n} {self.shape[0]}"
        data_size = self.bytes * n

        assert self.dtype.np_type == arr.dtype, f"{self.dtype.np_type}, {arr.dtype}"
        assert (arr.shape[0],) + self.shape[1:] == arr.shape, f"{self.shape}, {arr.shape}"

        ptr = arr.__array_interface__["data"][0]
        lib.cuda_upload(self.data_ptr, ptr, data_size)


    def download(self, n=None, device='numpy', stream=None):
        if device != 'numpy':
            import torch
            x = self.download(n, 'numpy')
            return torch.tensor(x, device=device)


        if n is not None:
            shape = (n,) + self.shape[1:]
            data_size = self.bytes * n
        else:
            shape = self.shape
            data_size = self.nbytes
        #shape = (n,) + self.shape[1:] if n is not None else self.shape
        arr = np.empty(shape, self.dtype.np_type)
        ptr = arr.__array_interface__["data"][0]

        if stream is not None:
            lib.cuda_download(ptr, self.data_ptr, data_size)
        else:
            lib.cuda_download_async(ptr, self.data_ptr, data_size, stream)
        return arr

    """
    def download_async(self, stream):
        arr = np.empty(self.shape, self.dtype.np_type)
        ptr = arr.__array_interface__["data"][0]
        lib.cuda_download_async(ptr, self.data_ptr, self.nbytes, stream)
        return arr
    """

    def copy_to_torch(self, tensor):
        assert tensor.device.type == "cuda"
        assert tensor.is_contiguous()
        width = spitch = self.dtype.size
        dpitch = tensor.stride()[0] * tensor.element_size()
        height = self.nbytes // width
        lib.cuda_copy2d(tensor.data_ptr(), dpitch, self.data_ptr, spitch, width, height)

    def cuda_add(self, values, stream=None):
        # TODO: make it faster
        x = self.download(n=len(values), stream=stream)
        x += values
        self.upload(x, stream)

    def zero(self, stream=None):
        if stream is None:
            lib.cuda_zero(self.data_ptr, self.nbytes)
        else:
            lib.cuda_zero_async(self.data_ptr, self.nbytes, stream)

    def zero_async(self, stream):
        lib.cuda_zero_async(self.data_ptr, self.nbytes, stream)

    def __del__(self):
        lib.cuda_free(self.data_ptr)

    def __repr__(self):
        return self.download().__repr__()