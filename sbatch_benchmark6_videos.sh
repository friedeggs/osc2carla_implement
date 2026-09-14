#!/bin/bash -l
#SBATCH --job-name=osc2-benchmark6-videos
#SBATCH --account=aip-six
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --chdir=/scratch/zwang179/traffic_orchestration/scenario_orchestrator_meta_repo/third_party/osc2runner
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Record CARLA videos for the six harness scenario families via osc2runner.
# Optional env (sbatch --export=ALL,ONLY=cut_in:lane_change:overtake ...):
#   ONLY=a:b:c       run a subset (use ':' — Slurm --export splits on ',')
#   SKIP_EXISTING=1  skip scenarios that already have an mp4
set -uo pipefail
export ONLY="${ONLY:-}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
exec ./run_record_benchmark6.sh
