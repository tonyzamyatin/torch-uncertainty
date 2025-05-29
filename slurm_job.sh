#SBATCH --job-name=anton_sweep
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate torch-uncertainty

python experiments/classification/mnist/run.py
