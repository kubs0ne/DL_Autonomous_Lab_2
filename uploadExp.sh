scp -r Experiments/1 nct01036@dt01.bsc.es:/home/nct01/nct01036/AutoLab2/Experiments #Upload the current experiment
ssh nct01036@plogin1.bsc.es 'sbatch AutoLab2/Experiments/1/ex1.sh' #Submit sbatch