from setuptools import setup

setup(name="scnn",
      version="0.1",
      description="Convolutional layer for cochains on simplicial complexes",
      author="Gard Spreemann",
      author_email="gard.spreemann@epfl.ch",
      license="GPL-3",
      packages=["scnn"],
      install_requires=["torch", "numpy", "scipy"])
