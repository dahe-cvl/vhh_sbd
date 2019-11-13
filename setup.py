try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
     name='vhh_sbd',
     version='0.1',
     author="Daniel Helm",
     author_email="daniel.helm@tuwien.ac.at",
     description="Shot Boundary Detection Package",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/dahe-cvl/vhh_sbd",
     packages=["sbd"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GPL License",
         "Operating System :: OS Independent",
     ],
 )