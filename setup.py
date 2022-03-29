try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='irasa',
      author="Joseph Rudoler",
      packages=find_packages(),
      version='0.1',
      py_modules=['irasa'],
      )
