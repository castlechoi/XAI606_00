lr_list=(1e-3 5e-3 5e-4 1e-2 1e-4 5e-5)

for lr_value in ${lr_list[@]}; do
    for LAYER in all $(seq 12 -1 0)
    do
        # CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
        python . probe -c ./configs/mert/MERT-v1-95M/VocalSetT.yaml \
        -o "optimizer.lr=${lr_value},,model.downstream_structure.components[0].layer='${LAYER}'"
        # "checkpoint.save_best_to=./best-layer-MERT-v1-95M/VocalSetT.ckpt"
    done
done