python3 ../../Tools/generate_input.py \
    -m MHM \
    -r 20 60 100 200 400 800 --cfl 0.95 \
    -o inputs-exp/ --max_step 10000000 \
    -i problem-inputs/eps1 -n eps1

python3 ../../Tools/generate_input.py \
    -m PESK-SP111 PESK-MUSCL-SSP222 PESK-MUSCL-SASSP322 \
    -r 20 60 100 200 400 800 --cfl 0.95 0.45 0.45 \
    -o inputs-IMEX/ --max_step 10000000 \
    -i problem-inputs/eps1 -n eps1

python3 ../../Tools/generate_input.py \
    -m MHM \
    -r 20 60 100 --cfl 0.95 \
    -o inputs-exp/ --max_step 10000000 \
    -i problem-inputs/eps1e-5 -n eps1e-5