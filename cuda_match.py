"""Batched 8-bit dot products through the NVIDIA driver shipped with Windows drivers.

The profile owns a private context and its cached template buffers. PTX is JIT
compiled by the installed driver; the end user does not need a CUDA toolkit.
"""
import ctypes as ct
import os
from weakref import finalize

import numpy as np


# One warp per (crop, reference). Every lane returns its exact integer subtotal.
# Summing the 32 lanes on the host also synchronizes the batch before classification.
_PTX = b"""
.version 6.0
.target sm_30
.address_size 64
.visible .entry dots(.param .u64 images, .param .u64 templates, .param .u64 output,
                     .param .u32 pixels, .param .u32 count) {
    .reg .pred done;
    .reg .u32 x, y, lane, i, n, refs, a, b, product;
    .reg .u64 im, tm, out, p, q, offset, total;
    ld.param.u64 im, [images];
    ld.param.u64 tm, [templates];
    ld.param.u64 out, [output];
    ld.param.u32 n, [pixels];
    ld.param.u32 refs, [count];
    mov.u32 x, %ctaid.x;
    mov.u32 y, %ctaid.y;
    mov.u32 lane, %tid.x;
    mad.wide.u32 im, y, n, im;
    mad.wide.u32 tm, x, n, tm;
    mov.u32 i, lane;
    mov.u64 total, 0;
loop:
    setp.ge.u32 done, i, n;
    @done bra finish;
    cvt.u64.u32 offset, i;
    add.u64 p, im, offset;
    add.u64 q, tm, offset;
    ld.global.u8 a, [p];
    ld.global.u8 b, [q];
    mad.wide.u32 total, a, b, total;
    add.u32 i, i, 32;
    bra loop;
finish:
    mad.lo.u32 i, y, refs, x;
    mad.lo.u32 i, i, 32, lane;
    mad.wide.u32 out, i, 8, out;
    st.global.u64 [out], total;
    ret;
}
"""


class CudaDot:
    def __init__(self):
        if os.name != 'nt':
            raise OSError('Windows NVIDIA driver required')
        self.driver = ct.WinDLL('nvcuda.dll', winmode=0x800)  # system32 only
        ptr, size, uint, address = ct.c_void_p, ct.c_size_t, ct.c_uint, ct.c_uint64
        for name, arguments in {
            'cuInit':[uint], 'cuDeviceGet':[ct.POINTER(ct.c_int),ct.c_int],
            'cuCtxCreate_v2':[ct.POINTER(ptr),uint,ct.c_int], 'cuCtxDestroy_v2':[ptr],
            'cuCtxPushCurrent_v2':[ptr], 'cuCtxPopCurrent_v2':[ct.POINTER(ptr)],
            'cuModuleLoadData':[ct.POINTER(ptr),ct.c_char_p],
            'cuModuleGetFunction':[ct.POINTER(ptr),ptr,ct.c_char_p],
            'cuMemAlloc_v2':[ct.POINTER(address),size], 'cuMemFree_v2':[address],
            'cuMemcpyHtoD_v2':[address,ptr,size], 'cuMemcpyDtoH_v2':[ptr,address,size],
            'cuLaunchKernel':[ptr,uint,uint,uint,uint,uint,uint,uint,ptr,ct.POINTER(ptr),ct.POINTER(ptr)],
        }.items():
            getattr(self.driver,name).argtypes = arguments
        self.call('cuInit',0)
        device, self.context = ct.c_int(), ptr()
        self.call('cuDeviceGet',ct.byref(device),0)
        self.call('cuCtxCreate_v2',ct.byref(self.context),4,device)  # blocking synchronization
        self.close = finalize(self,self.driver.cuCtxDestroy_v2,self.context)
        try:
            module, self.kernel = ptr(), ptr()
            self.call('cuModuleLoadData',ct.byref(module),_PTX)
            self.call('cuModuleGetFunction',ct.byref(self.kernel),module,b'dots')
            self.call('cuCtxPopCurrent_v2',ct.byref(ptr()))
        except BaseException:
            self.close()
            raise
        self.references = {}
        self.buffers = {}

    def call(self, name, *args):
        result = getattr(self.driver,name)(*args)
        if result:
            raise OSError(f'{name} failed (CUDA {result})')

    def allocate(self, size):
        address = ct.c_uint64()
        self.call('cuMemAlloc_v2',ct.byref(address),size)
        return address

    def buffer(self, key, size):
        address, capacity = self.buffers.get(key,(None,0))
        if size > capacity:
            if address is not None: self.call('cuMemFree_v2',address)
            address = self.allocate(size)
            self.buffers[key] = (address,size)
        return address

    def products(self, group, crops, references):
        inputs = np.ascontiguousarray(np.stack(crops))
        rows, cols = len(crops), len(references)
        partial = np.empty((rows,cols,32),dtype=np.uint64)
        self.call('cuCtxPushCurrent_v2',self.context)
        try:
            if group not in self.references:
                packed = np.ascontiguousarray(np.stack(references))
                address = self.allocate(packed.nbytes)
                self.call('cuMemcpyHtoD_v2',address,packed.ctypes.data,packed.nbytes)
                self.references[group] = address
            source = self.buffer('input',inputs.nbytes)
            dest = self.buffer('output',partial.nbytes)
            self.call('cuMemcpyHtoD_v2',source,inputs.ctypes.data,inputs.nbytes)
            args = (source,self.references[group],dest,ct.c_uint(crops[0].size),ct.c_uint(cols))
            pointers = (ct.c_void_p*len(args))(*(ct.addressof(arg) for arg in args))
            self.call('cuLaunchKernel',self.kernel,cols,rows,1,32,1,1,0,None,pointers,None)
            self.call('cuMemcpyDtoH_v2',partial.ctypes.data,dest,partial.nbytes)
        finally:
            self.call('cuCtxPopCurrent_v2',ct.byref(ct.c_void_p()))
        return partial.sum(axis=2,dtype=np.uint64)
