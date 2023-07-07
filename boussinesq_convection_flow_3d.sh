for i in 0 1 2 3 4 5 6 7 8 9
do
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python Boussinesq_convection_flow_3d.py --step_idx=$i
done