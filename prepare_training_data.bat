# export PYTHONPATH="./:$PYTHONPATH"
SET PYTHONPATH=%CD%;%PYTHONPATH%

python lib/data_utils/mpii3d_utils.py --dir ../Datasets/mpi_inf_3dhp