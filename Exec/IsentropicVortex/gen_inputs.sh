mkdir -p inputs_stat
mkdir -p inputs

python3 ../../Tools/generate_input.py -m HLLC MHM\
 EEEK-SP111 EEEK-MUSCL-SSP222 PESK-SP111 PESK-MUSCL-SSP222\
 -r 16 32 64 128 256 512 1024 --cfl 0.9 0.9 0.4\
 -n stat2d -i problem_inputs_stationary -o inputs_stat

python3 ../../Tools/generate_input.py -m HLLC MHM\
 EEEK-SP111 EEEK-MUSCL-SSP222 PESK-SP111 PESK-MUSCL-SSP222\
 -r 16 32 64 128 256 512 1024 --cfl 0.9 0.9 0.4\
 -n mov2d -i problem_inputs_moving -o inputs