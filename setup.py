from setuptools import setup

setup(name='semantic_segmentation',
      description='Provides scripts for creating/collecting a dataset for semantic segmentation, applying data augmentation and training/testing a semantic segmentation model.',
      version='0.1',
      author='Antonis Sidiropoulos',
      author_email='antosidi@ece.auth.gr',
      py_modules=['my_pkg'],
      install_requires=[
            'numpy',
            'opencv-python',
            'matplotlib',
            'pyrealsense2',
            'tqdm',
      ],
)
