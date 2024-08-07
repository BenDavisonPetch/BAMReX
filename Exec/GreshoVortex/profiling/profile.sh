#!/bin/bash
if [[ $# -lt 2 ]]; then
    echo "Usage: profile.sh profile_input executable [inputs] [output_dir]"
    echo "   profile_input is formatted as rows of num_ranks num_threads"
    exit 1
fi

if ! [[ -e $1 ]]; then
    echo "Cannot open file ${1}"
    exit 1
fi

if ! [[ -x $2 ]]; then
    echo "Cannot find executable ${2}"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ -z "$3" ]]; then
    INDIR=`realpath ${SCRIPT_DIR}/inputs/`
else
    INDIR=`realpath $3`
fi

if [[ ! -d ${INDIR} ]]; then
    echo "Cannot find input directory ${INDIR}"
    exit 1
fi

if [[ -z "$4" ]]; then
    OUTDIR=`realpath ${SCRIPT_DIR}/outputs`
else
    OUTDIR=`realpath $4`
fi

EXECUTABLE=`realpath ${2}`

read -r -a wcresult <<< `wc -l $1`
num_configs=${wcresult[0]}
num_inputs=`find inputs/ -type f | wc -l`

mkdir -p ${OUTDIR}

i=1
while read -u 4 line; do
    echo "Testing with config ${i}/${num_configs}..."
    read -r -a splitline <<< $line
    MPI_NUM_RANKS=${splitline[0]}
    export OMP_NUM_THREADS=${splitline[1]}
    echo "    MPI_NUM_RANKS=${MPI_NUM_RANKS} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    ((++i))
    echo "RUN 1/3"
    find ${INDIR} -type f -exec bash -c 'echo "    Starting test on input {}"; mpirun -np '${MPI_NUM_RANKS}' '${EXECUTABLE}' $0 amr.v=0 adv.v=0 amr.plot_files_output=0 > '${OUTDIR}'/$(basename $0).'${MPI_NUM_RANKS}'.${OMP_NUM_THREADS}.0; echo "    Done"' {} \; 
    echo "RUN 2/3"
    find ${INDIR} -type f -exec bash -c 'echo "    Starting test on input {}"; mpirun -np '${MPI_NUM_RANKS}' '${EXECUTABLE}' $0 amr.v=0 adv.v=0 amr.plot_files_output=0 > '${OUTDIR}'/$(basename $0).'${MPI_NUM_RANKS}'.${OMP_NUM_THREADS}.1; echo "    Done"' {} \; 
    echo "RUN 3/3"
    find ${INDIR} -type f -exec bash -c 'echo "    Starting test on input {}"; mpirun -np '${MPI_NUM_RANKS}' '${EXECUTABLE}' $0 amr.v=0 adv.v=0 amr.plot_files_output=0 > '${OUTDIR}'/$(basename $0).'${MPI_NUM_RANKS}'.${OMP_NUM_THREADS}.2; echo "    Done"' {} \; 
    echo "DONE"
done 4< $1