# Plugin package: Shot Boundary Detection

This package includes all methods to detect and split a given video into the basic shots. (currently focused on Abrupt Transitions).

## Package Description

PDF format: [vhh_sbd_pdf](https://github.com/dahe-cvl/vhh_sbd/blob/master/ApiSphinxDocumentation/build/latex/vhhpluginpackageshotboundarydetectionvhh_sbd.pdf)
    
HTML format (only usable if repository is available in local storage): [vhh_sbd_html](https://github.com/dahe-cvl/vhh_sbd/blob/master/ApiSphinxDocumentation/build/html/index.html)
    
    
## Quick Setup

**Requirements:**

   * Ubuntu 18.04 LTS (also tested on Windows 10)
   * python version 3.6.x
   
### 0 Environment Setup (optional)

**Create a virtual environment:**

   * create a folder to a specified path (e.g. /xxx/vhh_sbd/)
   * ```python3 -m venv /xxx/vhh_sbd/```

**Activate the environment:**

   * ```source /xxx/vhh_sbd/bin/activate```
   
### 1A Install using Pip

The VHH Shot Boundary Detection package is available on [PyPI](https://pypi.org/project/vhh-sbd/) and can be installed via ```pip```.

* Update pip and setuptools (tested using pip\==20.2.3 and setuptools==50.3.0)
* ```pip install vhh-sbd```

Alternatively, you can also build the package from source.

### 1B Install by building from Source

**Checkout vhh_sbd repository to a specified folder:**

   * ```git clone https://github.com/dahe-cvl/vhh_sbd```

**Install the sbd package and all dependencies:**

   * Update ```pip``` and ```setuptools``` (tested using pip\==20.2.3 and setuptools==50.3.0)
   * Install the ```wheel``` package: ```pip install wheel```
   * change to the root directory of the repository (includes setup.py)
   * ```python setup.py bdist_wheel```
   * The aforementioned command should create a /dist directory containing a wheel. Install the package using ```python -m pip install dist/xxx.whl```
   
> **_NOTE:_**
You can check the success of the installation by using the command *pip list*. This command should give you a list
with all installed python packages and it should include *vhh-sbd*.
   
   
### 2 Install PyTorch

Install a Version of PyTorch depending on your setup. Consult the [PyTorch website](https://pytorch.org/get-started/locally/) for detailed instructions.

### 3 Setup environment variables (optional)

   * ```source /data/dhelm/python_virtenv/vhh_sbd_env/bin/activate```
   * ```export CUDA_VISIBLE_DEVICES=1```
   * ```export PYTHONPATH=$PYTHONPATH:/XXX/vhh_sbd/:/XXX/vhh_sbd/Develop/:/XXX/vhh_sbd/Demo/```

### 4 Run demo script (optional)

   * change to root directory of the repository
   * ```python Demo/vhh_sbd_run_on_single_video.py```

> **_NOTE:_** 
Do not forget to change paths in demo script!


## Release Generation

* Create and checkout release branch: (e.g. v1.1.0): ```git checkout -b v1.1.0```
* Update version number in setup.py
* Update Sphinx documentation and release version
* Make sure that ```pip``` and ```setuptools``` are up to date
* Install ```wheel``` and ```twine```
* Build Source Archive and Built Distribution using ```python setup.py sdist bdist_wheel```
* Upload package to PyPI using ```twine upload dist/*```
