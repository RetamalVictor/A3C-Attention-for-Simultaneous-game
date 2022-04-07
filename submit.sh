
#!/bin/bash
#
#SBATCH --job-name=experiment-3-attention
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mail-user=victorrtml22@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --chdir="/home/baierh/tu-eind-AGSMCTS/"
#SBATCH --output=/home/baierh/outputs/test4-%j.out
#SBATCH --error=/home/baierh/outputs/test4-%j.err
#SBATCH --time=00:08:00


module load 2021
module load Python/3.9.5-GCCcore-10.3.0

source ./.research/bin/activate
cd ./tu-eind-AGSMCTS/src/learning/
python -u learning.py --nb-processes 16 --nb-players 4 --nb-soft-attention-heads 5 --hard-attention-rnn-hidden-size 128

