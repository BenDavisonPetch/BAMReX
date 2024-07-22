#!/bin/bash
MPI_COMMAND=mpirun

if [[ $# < 5 ]]; then
    echo "Usage: run.sh executable input_dir log_dir [working_dir] [log_ext]"
    exit 1
fi

EXECUTABLE=`realpath ${1}`
if [[ ! -x ${EXECUTABLE} ]]; then
    echo "Cannot find executable ${EXECUTABLE}"
    exit 1
fi
INDIR=`realpath ${2}`
if [[ ! -d ${INDIR} ]]; then
    echo "Input directory ${INDIR} does not exist"
    exit 1
fi
LOGDIR=`realpath ${3}`

if [[ ! -z $4 ]]; then
    if [[ ! -d $4 ]]; then
        echo "Working directory ${4} does not exist"
        exit 1
    fi
    cd ${4}
fi
LOG_EXT=$5

np=$SLURM_NTASKS

mkdir -p ${LOGDIR}

# mpirun just figures everything out from slurm environment variables
find ${INDIR} -type f -exec bash -c 'echo "Running {}"; '${MPI_COMMAND}' '${EXECUTABLE}' $0 amr.v=0 adv.v=0 > '${LOGDIR}'/$(basename $0).'${np}${LOG_EXT} {} \;

