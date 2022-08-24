import setuptools

from committee_learning  import (
  __pkgname__ as PKG_NAME,
  __author__  as AUTHOR
)

setuptools.setup(
  name = PKG_NAME,
  author  =  AUTHOR,
  packages = setuptools.find_packages(),
  python_requires = '>=3.7,<3.8', # Probably it works even with newr version of python, but still...
  install_requires = [
    'numpy',
    'torch',
    'sklearn',
    'matplotlib',
    'pyyaml',
    'tqdm',
  ]
)