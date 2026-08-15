#!/bin/bash -l
#SBATCH --job-name=osc2carla-closed-loop
#SBATCH --account=aip-six
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=sbatch-closed-loop-demo-%j.out
#SBATCH --error=sbatch-closed-loop-demo-%j.err
#
# Cluster-specific account/partition. Submit from this directory:
#   sbatch sbatch_closed_loop_demo.sh

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_record_closed_loop_demo.sh"
