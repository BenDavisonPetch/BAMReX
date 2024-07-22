EXEC_PATH=/local/data/public/bd431/BAMReX/Exec/IsentropicVortex
EXECUTABLE=/local/data/public/bd431/BAMReX/build/Exec/IsentropicVortex/isen_2d

#
# STATIONARY TESTS
#

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
mkdir -p ${SCRIPT_DIR}/logs

# Small tests (16, 32)
find ${EXEC_PATH}/inputs_stat/ -not -name "*1024" \
                          -not -name "*0512" \
                          -not -name "*0256" \
                          -not -name "*0128" \
                          -not -name "*0064" \
                          -not -type d \
                          -exec bash -c 'echo "    Starting test on input {}"; mpirun -np 1 '${EXECUTABLE}' $0 > '${SCRIPT_DIR}'/logs/$(basename $0); echo "    Done"' {} \;

# 64x64 tests
find ${EXEC_PATH}/inputs_stat/ -name "*0064" \
                          -exec bash -c 'echo "    Starting test on input {}"; mpirun -np 4 '${EXECUTABLE}' $0 > '${SCRIPT_DIR}'/logs/$(basename $0); echo "    Done"' {} \;

# 128x128 tests
find ${EXEC_PATH}/inputs_stat/ -name "*0128" \
                          -exec bash -c 'echo "    Starting test on input {}"; mpirun -np 16 '${EXECUTABLE}' $0 > '${SCRIPT_DIR}'/logs/$(basename $0); echo "    Done"' {} \;

# All others
find ${EXEC_PATH}/inputs_stat/ -not -name "*0016" \
                          -not -name "*0032" \
                          -not -name "*0064" \
                          -not -name "*0128" \
                          -not -type d \
                          -exec bash -c 'echo "    Starting test on input {}"; mpirun -np 32 '${EXECUTABLE}' $0 > '${SCRIPT_DIR}'/logs/$(basename $0); echo "    Done"' {} \;