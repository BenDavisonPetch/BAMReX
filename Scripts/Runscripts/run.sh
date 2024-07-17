if [[ $# < 1 ]]; then
    echo "Usage: run.sh MPI_NUM_RANKS [input_dir] [log_dir] [executable]"
    exit 1
fi

MPI_NUM_RANKS=$1
if [[ -z "$2" ]]; then
    INDIR=`realpath ./inputs`
else
    INDIR=`realpath ${2}`
fi

if [[ ! -d ${INDIR} ]]; then
    echo "Input directory ${INDIR} does not exist"
    exit 1
fi

if [[ -z "$3" ]]; then
    LOGDIR=`realpath ./logs`
else
    LOGDIR=`realpath ${3}`
fi

if [[ -z "$4" ]]; then
    EXECUTABLE=`realpath ./gresho_2d`
else
    EXECUTABLE=`realpath ${4}`
fi

if [[ ! -x ${EXECUTABLE} ]]; then
    echo "Cannot find executable ${EXECUTABLE}"
    exit 1
fi

mkdir -p ${LOGDIR}

find ${INDIR} -type f -exec bash -c 'echo "Starting test {}"; mpirun -np '${MPI_NUM_RANKS}' '${EXECUTABLE}' $0 amr.v=0 adv.v=0 > '${LOGDIR}'/$(basename $0).'${MPI_NUM_RANKS}'; echo "    Done"' {} \;

