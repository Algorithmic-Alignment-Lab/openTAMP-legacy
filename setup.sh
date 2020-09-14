FULL_INSTALL=true
cd ~
mkdir tamp_work
cd ~/tamp_work



### SETUP VIRTENV ###
# Setup virtual env
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
echo 'alias tampenv="source ~/tamp_work/venv/bin/activate"' >> ~/.bashrc

# Get python code
pip install numpy==1.18.5
pip install seaborn numba pandas pybullet dm_control numdifftools ipdb
pip install --force-reinstall numpy==1.18.5



### SETUP TAMP CODE ###
git clone https://github.com/dhadfieldmenell/tampy.git
git clone https://github.com/m-j-mcdonald/sco.git
cd tampy
git checkout python3 
git pull origin python3 
cd ../sco
git checkout python3
pip install h5py psutil
pip install --upgrade numpy
cd ~/tamp_work

# Set env variables
echo 'export GUROBI_HOME=/home/${USER}/gurobi702/linux64' >> ~/.bashrc
echo 'export PATH="${PATH}:${GUROBI_HOME}:${GUROBI_HOME}/bin":/home/${USER}/.local/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:/home/${USER}/.mujoco/mujoco200/bin"' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/home/${USER}/tamp_work/sco:/home/${USER}/tamp_work/tampy/src' >> ~/.bashrc

# Setup gurobi (need to get license separately)
# https://www.gurobi.com/downloads/end-user-license-agreement-academic/
cd ~/tamp_work
wget https://packages.gurobi.com/7.0/gurobi7.0.2_linux64.tar.gz
tar xvfz gurobi7.0.2_linux64.tar.gz 
cd gurobi702/linux64

source ~/.bashrc
python setup.py install



### SETUP POLICY TRAINING CODE ###
cd ~
# Setup mujoco
# Place your mujoco key in your home directory
mkdir .mujoco
cd .mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco200.zip
unzip mujoco200.zip
cp ~/mjkey.txt .

# Setup additional codebases
cd ~/tamp_work
git clone https://github.com/m-j-mcdonald/BaxterGym.git
git clone https://github.com/m-j-mcdonald/gps.git
pip install tensorflow==1.10.0
pip install -e BaxterGym
echo 'export MUJOCO_KEY_PATH=/home/${USER}/.mujoco/mjkey.txt' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/home/${USER}/tamp_work/gps/python' >> ~/.bashrc

pip install baselines
pip install --force-reinstall numpy==1.18.5
