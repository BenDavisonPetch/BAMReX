BAMReX_DIR = "/rds/user/bd431/hpc-work/BAMReX"

executable = BAMReX_DIR+"/build/Exec/GreshoVortex/gresho_2d"
runscript = BAMReX_DIR+"/Scripts/Runscripts/run_openmpi.sh"

input_dir = BAMReX_DIR+"/Exec/GreshoVortex/profiling/inputs-PESK-multires"
log_dir = BAMReX_DIR+"/build/Exec/GreshoVortex/profiling/outputs-pesk-mgs64-multires"
work_dir = BAMReX_DIR+"/build/Exec/GreshoVortex"

project = "NIKIFORAKIS-SL3-CPU"

file_fmt = "slurm_submit.pesk_mgs64_multires.{_N_MPI_TASKS}"
jobname_fmt = "pesk.{_N_MPI_TASKS}"

# Formatted as [[n_nodes, n_tasks]]
sizes = [[1, 1],
         [1, 2],
         [1, 4],
         [1, 8],
         [1, 16],
         [1, 32],
         [1, 64],
         [2, 128]]

runtimes = ["03:00:00",
            "02:00:00",
            "01:30:00",
            "01:00:00",
            "00:45:00",
            "00:30:00",
            "00:15:00",
            "00:15:00"]

script = r"""#!/bin/bash
#!
#! Example SLURM job script for Peta4-IceLake (Ice Lake CPUs, HDR200 IB)
#! Last updated: Sat Jul 31 15:39:45 BST 2021
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J {_JOB_NAME}
#! Which project should be charged:
#SBATCH -A {_PROJECT}
#SBATCH -p icelake
#! How many whole nodes should be allocated?
#SBATCH --nodes={_N_NODES}
#! How many (MPI) tasks will there be in total? (<= nodes*76)
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and
#! 3380 MiB of memory per CPU.
#SBATCH --ntasks={_N_MPI_TASKS}
#! How much wallclock time will be required?
#SBATCH --time={_RUNTIME}
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by cpu number*walltime.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*76 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 CPU by default, and each CPU is allocated 3380 MiB
#! of memory. If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MiB per node).

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
#module load rhel8/default-icl              # REQUIRED - loads the basic environment
module load openmpi-4.0.5-gcc-8.4.1-l7ihwk3
module load rhel8/slurm

#! Insert additional module load commands after this line if needed:

#! Full path to application executable: 
application=""

#! Run options for the application:
options=""

#! Work directory (i.e. where the job will run):
#workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.
workdir={_WORK_DIR}

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 76:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${{numnodes}}*${{mpi_tasks_per_node}}]

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
#! Notes:
#! 1. These variables influence Intel MPI only.
#! 2. Domains are non-overlapping sets of cores which map 1-1 to MPI tasks.
#! 3. I_MPI_PIN_PROCESSOR_LIST is ignored if I_MPI_PIN_DOMAIN is set.
#! 4. If MPI tasks perform better when sharing caches/sockets, try I_MPI_PIN_ORDER=compact.


#! Uncomment one choice for CMD below (add mpirun/mpiexec options if necessary):

#! Choose this for a MPI code (possibly using OpenMP) using Intel MPI.
#CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
#CMD="$application $options"

#! Choose this for a MPI code (possibly using OpenMP) using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"

CMD="{_RUNSCRIPT_LOC} {_EXECUTABLE} {_INPUT_DIR} {_LOG_DIR} $workdir .0 \
    && {_RUNSCRIPT_LOC} {_EXECUTABLE} {_INPUT_DIR} {_LOG_DIR} $workdir .1 \
    && {_RUNSCRIPT_LOC} {_EXECUTABLE} {_INPUT_DIR} {_LOG_DIR} $workdir .2"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 

"""

assert(len(sizes) == len(runtimes))
for i in range(len(sizes)):
    filename = file_fmt.format(_N_MPI_TASKS=sizes[i][1])
    jobname = jobname_fmt.format(_N_MPI_TASKS=sizes[i][1])
    filecontents = script.format(_N_MPI_TASKS=sizes[i][1],
                                 _N_NODES=sizes[i][0],
                                 _RUNTIME=runtimes[i],
                                 _RUNSCRIPT_LOC=runscript,
                                 _EXECUTABLE=executable,
                                 _INPUT_DIR=input_dir,
                                 _LOG_DIR=log_dir,
                                 _WORK_DIR=work_dir,
                                 _JOB_NAME=jobname,
                                 _PROJECT=project)
    with open(filename, "w") as outfile:
        outfile.write(filecontents)
