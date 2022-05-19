scp -r Experiments/3 nct01036@dt01.bsc.es:/home/nct01/nct01036/AutoLab2/Experiments #Upload the current experiment
ssh nct01036@plogin1.bsc.es 'sbatch AutoLab2/Experiments/3/ex3.sh' #Submit sbatch
