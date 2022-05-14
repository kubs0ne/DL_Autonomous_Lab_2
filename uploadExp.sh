scp -r Experiments/2 nct01029@dt01.bsc.es:/home/nct01/nct01029/AutoLab2/Experiments #Upload the current experiment
ssh nct01029@plogin1.bsc.es 'sbatch AutoLab2/Experiments/2/ex2.sh' #Submit sbatch
