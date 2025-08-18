#!/bin/bash

# Ensemble size
n=100

# Number of EKI iterations
counter=$1
if [ -z $counter ]
then
    echo "Don't forget counter!"
    exit
fi

# Jobs
gcm=$(sbatch --parsable --array=1-$n run_eki)
eki=$(sbatch --parsable --depend=afterok:$gcm run_main)

# Wait until jobs are done
while true; do
    sleep 5

    x=$(qstat | grep $eki | awk '{print $5}')
    if [ $x == "C" ]
    then
	break
    fi

    if [ -z $x ]
    then
	echo 'NOTHING IN QSTAT'
	exit
    fi

    x=$(squeue | grep $eki | awk '{print $8}')
    if [ $x == "(DependencyNeverSatisfied)" ] 
    then
	echo "GCM iteration never completed!"
	echo "Job ID "$eki
	scancel $eki
	exit
    fi
        
done

# Rerun script until done with all iterations
if [ $counter -gt 1 ]
then
    counter=$(echo $counter-1 | bc)
    echo $counter
    ./run_eki.sh $counter
fi
