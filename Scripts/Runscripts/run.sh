#!/bin/bash
MPI_COMMAND=mpirun

if [[ $# < 5 ]]; then
    echo "Usage: run.sh mpi_tasks_per_node np executable input_dir log_dir [working_dir] [log_ext]"
    exit 1
fi

mpi_tasks_per_node=$1
np=$2
EXECUTABLE=`realpath ${3}`
if [[ ! -x ${EXECUTABLE} ]]; then
    echo "Cannot find executable ${EXECUTABLE}"
    exit 1
fi
INDIR=`realpath ${4}`
if [[ ! -d ${INDIR} ]]; then
    echo "Input directory ${INDIR} does not exist"
    exit 1
fi
LOGDIR=`realpath ${5}`

if [[ ! -z $6 ]]; then
    if [[ ! -d $6 ]]; then
        echo "Working directory ${6} does not exist"
        exit 1
    fi
    cd ${6}
fi
LOG_EXT=$7

mkdir -p ${LOGDIR}

find ${INDIR} -type f -exec bash -c 'echo "Running {}"; '${MPI_COMMAND}' -ppn '${mpi_tasks_per_node}' -np '${np}' '${EXECUTABLE}' $0 amr.v=0 adv.v=0 > '${LOGDIR}'/$(basename $0).'${np}${LOG_EXT} {} \;

