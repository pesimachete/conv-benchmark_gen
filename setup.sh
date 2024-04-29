CONDA_HOME=~/conda-test
CONDA=$CONDA_HOME/bin/conda

NEURALSAT_CONDA_HOME=$CONDA_HOME/envs/benchmark-neuralsat
NEURALSAT_PY=$NEURALSAT_CONDA_HOME/bin/python

CROWN_CONDA_HOME=$CONDA_HOME/envs/benchmark-crown
CROWN_PY=$CROWN_CONDA_HOME/bin/python

BENCHMARK_HOME=$(pwd)

# # Install NVIDIA driver
# wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.54.03/NVIDIA-Linux-x86_64-535.54.03.run

# sudo nvidia-smi -pm 0
# chmod +x ./NVIDIA-Linux-x86_64-535.54.03.run
# sudo ./NVIDIA-Linux-x86_64-535.54.03.run --silent --dkms
# # Remove old driver (if already installed) and reload the new one.
# sudo rmmod nvidia_uvm; sudo rmmod nvidia_drm; sudo rmmod nvidia_modeset; sudo rmmod nvidia
# sudo modprobe nvidia; sudo nvidia-smi -e 0; sudo nvidia-smi -r -i 0
# sudo nvidia-smi -pm 1
# # Make sure GPU shows up.
# nvidia-smi


# step 1: install conda
if [ ! -d $CONDA_HOME ]; then
    wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
    chmod 755 conda.sh
    ./conda.sh -bf -p $CONDA_HOME
    rm ./conda.sh
fi

# step 2: install environments
if [ ! -d $NEURALSAT_CONDA_HOME ]; then
    $CONDA env create -p $NEURALSAT_CONDA_HOME -f neuralsat.yaml
fi  

if [ ! -d $CROWN_CONDA_HOME ]; then
    $CONDA env create -p $CROWN_CONDA_HOME -f crown.yaml
fi

# step 3: install verifiers
# echo $BENCHMARK_HOME
mkdir -p $BENCHMARK_HOME/tools/

if [ ! -d $BENCHMARK_HOME/tools/neuralsat ]; then
    git clone https://github.com/dynaroars/neuralsat $BENCHMARK_HOME/tools/neuralsat
fi

if [ ! -d $BENCHMARK_HOME/tools/alpha-beta-CROWN ]; then
    git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN $BENCHMARK_HOME/tools/alpha-beta-CROWN
fi

# step 4: export variables
export NEURALSAT_PY=$NEURALSAT_PY
export CROWN_PY=$CROWN_PY
