import codecs
import os.path

from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext as build_ext_orig

#  https://stackoverflow.com/questions/4529555/building-a-ctypes-based-c-library-with-distutils
class build_ext(build_ext_orig):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, Extension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)

"""Reading version as proposed in https://packaging.python.org/guides/single-sourcing-package-version/"""
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# Generate Python library
setup(
    name="PyMKL",
    description="Multiple Kernel Learning implementation for Python",
    url="https://github.com/gjimenez/PyMKL",
    author="Guillermo Jimenez-Perez",
    author_email="<guillermo@jimenezperez.com>",
    # Needed to actually package something
    packages=find_packages(),
    # Needed for dependencies
    install_requires=["numpy", "scipy", "picos", "cvxopt", "smcp", "joblib", 
                      "pandas", "tqdm", "scikit-learn", "networkx",
                      "sak @ git+https://github.com/guillermo-jimenez/sak.git"],
    # *strongly* suggested for sharing
    version=get_version("PyMKL/__init__.py"),
    # The license can be anything you like
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules = ["libPyMKL"],
    ext_modules=[
        Extension(
            "PyMKL/lib/libPyMKL",
            ["PyMKL/lib/libPyMKL.c",],
        ),
    ],
    cmdclass={'build_ext': build_ext},
)