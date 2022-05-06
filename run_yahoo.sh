#!/bin/sh
# The folliwing lines instruct Slurm to allocate one GPU.
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


source $PATH_TO_CONDA.sh
conda activate bine
python ./src/utils/bine.py

for i in `seq 25 25 125`; do 
    python BiNE/model/train.py --train-data ./data/processed/bine_embed_data.dat --d $i  --test-data ./data/processed/bine_embed_data.dat --lam 0.025 --max-iter 50 --model-name yahoo --rec 1 --large 2 --vectors-u ./models/embeddings/vectors_u.dat --vectors-v ./models/embeddings/vectors_v.dat
    sbatch -W ultr-pref-elicit_yahoo.job
    sbatch -W ultr-pref-elicit_meta_yahoo.job
    mv ./logs/pref_elicit_main* ./reports/figures
    if [ -d "$RESULTS" ]
    then
        rm -r "$RESULTS"
        mkdir "$RESULTS"
    else
        printf '%s\n' "logs dir. exists. Creating new one ($RESULTS)"
        mkdir "$RESULTS"
    fi
done



