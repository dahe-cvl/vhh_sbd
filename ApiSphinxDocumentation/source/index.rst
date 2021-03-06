.. vhh_sbd documentation master file, created by
   sphinx-quickstart on Wed May  6 18:41:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Package Overview
================

The following list give an overview of the folder structure of this python repository:

*name of repository*: vhh_sbd

   * **ApiSphinxDocumentation/**: includes all files to generate the documentation as well as the created documentations
                                  (html, pdf)
   * **config/**: this folder includes the required configuration file
   * **sbd/**: this folder represents the shot-type-classification module and builds the main part of this repository
   * **Demo/**: this folder includes a demo script to demonstrate how the package have to be used in customized applications
   * **Develop/**: includes scripts to evaluate the implemented approach as well as several helper scripts used during
                   development stage. Furthermore, a script is included to create the package documentation (pdf, html).
   * **README.md**: this file gives a brief description of this repository (e.g. link to this documentation)
   * **requirements.txt**: this file holds all python lib dependencies and is needed to install the package in your own
                           virtual environment
   * **setup.py**: this script is needed to install the sbd package in your own virtual environment

Setup  instructions
===================

This package includes a setup.py script and a requirements.txt file which are needed to install this package for custom
applications. The following instructions have to be done to used this library in your own application:

Requirements:

   * Ubuntu 18.04 LTS
   * CUDA 10.1 + cuDNN
   * python version 3.6.x

Create a virtual environment:

   * create a folder to a specified path (e.g. /xxx/vhh_sbd/)
   * python3 -m venv /xxx/vhh_sbd/

Activate the environment:

   * source /xxx/vhh_sbd/bin/activate

Checkout vhh_sbd repository to a specified folder:

   * git clone https://github.com/dahe-cvl/vhh_sbd

Install the sbd package and all dependencies:

   * change to the root directory of the repository (includes setup.py)
   * python setup.py install

.. note::

  You can check the success of the installation by using the commend *pip list*. This command should give you a list with all installed python packages and it should include *vhh_sbd*

.. note::

   Currently there is an issue in the *setup.py* script. Therefore the pytorch libraries have to be installed manually by running the following command:
   *pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html*

Parameter Description
=====================

.. autoyaml:: ../config/config_vhh_test.yaml

API Description
===============

This section gives an overview of all classes and modules in *sbd* as well as an code description.

.. toctree::
   :maxdepth: 4

   Configuration.rst
   CandidateSelection.rst
   Evaluation.rst
   PreProcessing.rst
   Models.rst
   SBD.rst
   Shot.rst
   Video.rst
   Utils.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

**********
References
**********
.. target-notes::

.. _`Visual History of the Holocaust`: https://www.vhh-project.eu/
.. _`vhh_core`: https://github.com/dahe-cvl/vhh_core
.. _`vhh_sbd`: https://github.com/dahe-cvl/vhh_sbd

