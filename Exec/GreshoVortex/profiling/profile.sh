#!/bin/bash
if [[ $# -lt 2 ]]; then
    echo "Usage: profile.sh profile_input executable"
    echo "   profile_input is formatted as rows of num_ranks num_threads"
    exit 1
fi

if ! [[ -e $1 ]]; then
    echo "Cannot open file ${1}"
    exit 1
fi

if ! [[ -e $2 ]]; then
    echo "Cannot find executable ${2}"
    exit 1
fi

EXECUTABLE=`realpath ${2}`

read -r -a wcresult <<< `wc -l $1`
num_configs=${wcresult[0]}
num_inputs=`find inputs/ -type f | wc -l`

mkdir -p outputs

i=1
while read -u 4 line; do
    echo "Testing with config ${i}/${num_configs}..."
    read -r -a splitline <<< $line
    MPI_NUM_RANKS=${splitline[0]}
    export OMP_NUM_THREADS=${splitline[1]}
    echo "    MPI_NUM_RANKS=${MPI_NUM_RANKS} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    ((++i))
    find inputs/ -type f -exec bash -c 'mpirun -np '${MPI_NUM_RANKS}' '${EXECUTABLE}' $0 > outputs/$(basename $0).'${MPI_NUM_RANKS}'.${OMP_NUM_THREADS}' {} \; 
done 4< $1