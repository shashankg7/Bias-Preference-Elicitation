ULTR_PrefElicit
==============================

Code for anonymous ICTIR Submission for the paper, "A First Look at Selection Bias in Preference Elicitation for Recommendation". 

Paper-ID for the submissions: 39

## Installing Conda Env for Synthetic Topic Simulation
To generate synthetic topics for items, we create a user-item graph and use graph embeddings, using [BiNE method](http://staff.ustc.edu.cn/~hexn/papers/sigir18-bipartiteNE.pdf). 

To run the BiNE network embedding, create an conda env using the "req_bine.txt" file provided with the repo.

$conda create -n <environment-name> --file req_bine.txt

## Installing Conda Env for the Main code
For running the main code, create a conda env using the "req.txt" file provided with the repo. 

$conda create -n <environment-name> --file req.txt
## Steps to run the simulator:

### First step is to download the dataset, from here
- download the [Yahoo! R3 dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r). 
- download the [Coat dataset](https://www.cs.cornell.edu/~schnabts/mnar/)
  
Put them in the ./data/raw/yahoo, and ./data/raw/coat folders respectively.

The python file ./src/simulator.py has the code to generate the two fully-synthetic datasets (from coat and yahoo), and one fully-synthetic dataset.

$python src/simulator.py --data coat

The dataset is generated at line #73, #86, and #97 in the code. Following that the dataset can be used for different downstream tasks. 

### MAIN RUN SCRIPTS

To run the script for different datasets, run the following script

$ sbatch run_$DATASET.sh