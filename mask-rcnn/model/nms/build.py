import os
import torch
from torch.utils.ffi import create_extension
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, include_paths


sources = ['src/nms.c']
headers = ['src/nms.h']
defines = []
with_cuda = False

# if torch.cuda.is_available():
#     print('Including CUDA code.')
#     sources += ['src/nms_cuda.c']
#     headers += ['src/nms_cuda.h']
#     defines += [('WITH_CUDA', None)]
#     with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
# extra_objects = ['src/cuda/nms_kernel.cu.o']
extra_objects = ['C:\\Project\\floor-sp\\venv\\Lib\\site-packages\\torch\\lib\\ATen.lib']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]


# setup(
#     name='extension',
#     ext_modules=[
#         Extension(
#             name='extension',
#             sources=sources,
#             extra_objects=extra_objects,
#             include_dirs=include_paths(),
#             extra_compile_args=['-g'],
#             language='c',),
#
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
# })

ffi = create_extension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
