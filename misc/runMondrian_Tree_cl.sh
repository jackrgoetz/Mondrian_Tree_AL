#PBS -N Mondrian_Tree_cl
#PBS -M jrgoetz@umich.edu
#PBS -m abe
#PBS -A tewaria_fluxm
#PBS -q fluxm
#PBS -l qos=flux
#PBS -l nodes=1:ppn=1,walltime=20:00:00,pmem=4gb
#PBS -V

# Set output and error directories
#PBS -j oe
#Command to execute MATLAB program

cd ${PBS_O_WORKDIR}

python paper_cl_forest.py