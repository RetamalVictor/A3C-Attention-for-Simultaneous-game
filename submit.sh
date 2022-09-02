
#!/bin/bash
#
#SBATCH --job-name=<Exp name>
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mail-user=<mail here>
#SBATCH --mail-type=ALL
#SBATCH --chdir="<your pwd here>"
#SBATCH --output=<out path>
#SBATCH --error=<error path>
#SBATCH --time=00:08:00


module load 2021
module load Python/3.9.5-GCCcore-10.3.0

source ./.research/bin/activate
cd ./A3C-Attention-for-Simultaneous-game/src/learning
python -u learning.py --nb-processes 16 --nb-players 4 --nb-soft-attention-heads 5 --hard-attention-rnn-hidden-size 128

