ULTR_PrefElicit
==============================

Code for Unbiased learning to rank for preference elicitation

## Steps to run the code:

### For yahoo dataset, train BiNE model
sbatch bine_ui.job

Run this one time only, then run following commands:

1) bash run.sh
2) sbatch pref_elicit_meta.job