python3 ../../Tools/generate_input.py \
    -m HLLC MHM PESK-SP111 PESK-MUSCL-SSP222 PESK-MUSCL-SASSP332 \
    -r 20 50 100 200 400 800 --cfl 0.95 0.95 0.95 0.45 0.45 \
    -o inputs-order2/ --max_step 10000000 \
    -T ../../Tools/input_templates/no_bf_template \
    -i problem-inputs-eps1 -n eps1

python3 ../../Tools/generate_input.py \
    -m HLLC MHM PESK-SP111 PESK-MUSCL-SSP222 PESK-MUSCL-SASSP332 \
    -r 20 50 100 200 400 800 --cfl 0.95 0.95 0.45 0.45 \
    -o inputs-order2/ --max_step 10000000 \
    -T ../../Tools/input_templates/no_bf_template \
    -i problem-inputs-eps1e-5 -n eps1e-5

python3 ../../Tools/generate_input.py \
    -m PESK-SSP222 PESK-SASSP332 \
    -r 20 50 100 200 400 800 --cfl 0.95 \
    -o inputs-order2/ --max_step 10000000 \
    -T ../../Tools/input_templates/no_bf_template \
    -i problem-inputs-eps1 -n eps1