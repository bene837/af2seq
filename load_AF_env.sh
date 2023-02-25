#Execute this file using the 'source' command from your terminal!
#
# Load the correct dependencies
echo Purging modules...
module purge
echo Loading modules:
echo gcc 8.4.0-cuda...
module load gcc/8.4.0-cuda
echo cuda 11.0.2...
module load cuda/11.0.2
echo cudnn 8.0.2.39-11.0-linux-x64...
module load cudnn/8.0.2.39-11.0-linux-x64
echo Modules succesfully loaded!

# Load anaconda and load the virtual environment with correct packages
echo Setting up virtual python environment...
source /home/goverde/anaconda3/etc/profile.d/conda.sh
conda activate /home/goverde/anaconda3/envs/AF2

# Check the python version
echo Running on python version:
python --version

# Check if CUDA device is available for pytorch
echo Checking CUDA device:
python check_cuda_tf.py
