#!/bin/bash

#SBATCH --job-name=650project_vec-map

#SBATCH --time=01:00:00

#SBATCH --account=si650f25s001_class

#SBATCH --partition=standard

#SBATCH --nodes=1

#SBATCH --cpus-per-task=2

#SBATCH --ntasks-per-node=1

#SBATCH --mem=64g

#SBATCH --mail-type=BEGIN,END

#SBATCH --mail-user=yixuanch@umich.edu

#SBATCH --output=/home/yixuanch/650project/scripts/greatlakes/vec-map/logs/%x-%j.log

BASE_PATH=/home/yixuanch/650project/scripts/greatlakes/vec-map

# Create logs directory if it doesn't exist
mkdir -p ${BASE_PATH}/logs

# Install required dependencies
pip install -r ${BASE_PATH}/requirements.txt

# Change to the vec-map directory
cd ${BASE_PATH}

# Force unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
python -u ...