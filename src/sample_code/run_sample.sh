#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sample_out.log
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding    
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load the numactl module to enable numa library linking
module load numactl

# Compile
gcc -O3 -lm -lnuma --openmp main.c -o main

mkdir -p ../test_images_out

# Run
srun  ./main valve.png valve-out.png 128
srun  ./main ../test_images/720x480.png ../test_images_out/720x480-out.png 128
srun  ./main ../test_images/1024x768.png ../test_images_out/1024x768-out.png 128
srun  ./main ../test_images/1920x1200.png ../test_images_out/1920x1200-out.png 128
srun  ./main ../test_images/3840x2160.png ../test_images_out/3840x2160-out.png 128
srun  ./main ../test_images/7680x4320.png ../test_images_out/7680x4320-out.png 128

