# Plugin package: Shot Boundary Detection

This package includes all methods to detect and split a given video into the basic shots. (currently focused on Abrupt Transitions).

## Package Description

PDF format: [vhh_sbd_pdf](https://github.com/dahe-cvl/vhh_sbd/blob/master/ApiSphinxDocumentation/build/latex/vhhpluginpackageshotboundarydetectionvhh_sbd.pdf)
    
HTML format (only usable if repository is available in local storage): [vhh_sbd_html](https://github.com/dahe-cvl/vhh_sbd/blob/master/ApiSphinxDocumentation/build/html/index.html)
    
    
## Quick Setup

**Requirements:**

   * Ubuntu 18.04 LTS
   * python version 3.6.x

**Create a virtual environment:**

   * create a folder to a specified path (e.g. /xxx/vhh_sbd/)
   * python3 -m venv /xxx/vhh_sbd/

**Activate the environment:**

   * source /xxx/vhh_sbd/bin/activate

**Checkout vhh_sbd repository to a specified folder:**

   * git clone https://github.com/dahe-cvl/vhh_sbd

**Install the cmc package and all dependencies:**

   * change to the root directory of the repository (includes setup.py)
   * python setup.py install

**Setup environment variables:**

   * source /data/dhelm/python_virtenv/vhh_sbd_env/bin/activate
   * export CUDA_VISIBLE_DEVICES=1
   * export PYTHONPATH=$PYTHONPATH:/XXX/vhh_sbd/:/XXX/vhh_sbd/Develop/:/XXX/vhh_sbd/Demo/


> **_NOTE:_**
  You can check the success of the installation by using the commend *pip list*. This command should give you a list
  with all installed python packages and it should include *vhh_sbd*.

**Run demo script**

   * change to root directory of the repository
   * python Demo/vhh_sbd_run_on_single_video.py

e.g. python Demo/vhh_sbd_on_single_video.py /data/share/maxrecall_vhh_mmsi/develop/videos/downloaded/3.m4v 
            /home/dhelm/VHH_Develop/installed_pkg/vhh_pkgs/vhh_sbd/config/config_vhh_test.yaml


