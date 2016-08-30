# -*- coding:utf-8 -*-
from distutils.core import setup,Extension
import os
import sys
'''
compile_args = ['-std=c++11','-funroll-loops','-O3']
f sys.platform == 'darwin':
    compile_args.extend(["-stdlib=libc++",'-mmacosx-version-min=10.7'])

package_data = {}
fasttext_dir = "fastText/src/"
module = [Extension('fastTextPy/libfasttext',
                        include_dirs=['fastText/src'],
                        libraries=['pthread'],
                        sources=[
                                fasttext_dir+'py.cpp',
                                fasttext_dir+'args.cc',
                                fasttext_dir+'dictionary.cc',
                                fasttext_dir+'matrix.cc',
                                fasttext_dir+'model.cc',
                                fasttext_dir+'utils.cc',
                                fasttext_dir+'vector.cc'],
                        extra_compile_args=compile_args,
                        language='c++'),]
'''
if len(sys.argv)>1 and sys.argv[1] in ("build",'install'):
    if os.system("cd fastText && make && "
             "mv libfasttext.so ../fastTextPy && make clean")!=0:
        raise RuntimeError("can't compile module fasttext")


setup(name='fastTextPy',
      version='0.1.0',
      description='Python library for fasttext',
      author='frank Lee',
      author_email='golifang123@gmail.com',
      #ext_modules = module,
      packages=['fastTextPy'],
      package_data={"fastTextPy":['*.so']},
      url='https://github.com/mklf/fastText',
      install_requires=[
          'numpy >=1.3',
          'scipy>=0.7.0',
      ]
)
