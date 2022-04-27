#!/bin/sh
# The folliwing lines instruct Slurm to allocate one GPU.
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=s.gupta2@uva.nl # Where to send mail	
#SBATCH -o ./logs/%A_%a.out                     
#SBATCH -e ./logs/%A_%a.err                                                         
#SBATCH --mem=50G
#SBATCH --time=25:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

LOGS=./logs
RESULTS=reports/results
if [ -d "$LOGS" ]
then
    printf '%s\n' "Removing logs ($LOGS)"
    rm -r "$LOGS"
    mkdir "$LOGS"
else
    printf '%s\n' "logs dir. doesn't exist. Creating new one ($LOGS)"
    mkdir "$LOGS"
fi

if [ -d "$RESULTS" ]
then
    rm -r "$RESULTS"
    mkdir "$RESULTS"
else
    printf '%s\n' "logs dir. exists. Creating new one ($RESULTS)"
    mkdir "$RESULTS"
fi

sbatch -W ultr-pref-elicit_coat.job
sbatch -W ultr-pref-elicit_meta_coat.job
mv ./logs/pref_elicit_main* ./reports/figures



