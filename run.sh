#!/bin/bash

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

sbatch ultr-pref-elicit.job
#sbatch ultr-pref-elicit_meta.job
