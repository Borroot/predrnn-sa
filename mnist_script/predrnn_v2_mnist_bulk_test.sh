export CUDA_VISIBLE_DEVICES=0
cd ..
for n in $(seq 500 500 30000)
do
    # get the test results
    python -u run.py \
        --is_training 0 \
        --device cuda \
        --dataset_name mnist \
        --train_data_paths data/moving-mnist-example/moving-mnist-train.npz \
        --valid_data_paths data/moving-mnist-example/moving-mnist-valid.npz \
        --save_dir checkpoints/mnist_predrnn_v2 \
        --gen_frm_dir results/mnist_predrnn_v2 \
        --model_name predrnn_v2 \
        --reverse_input 1 \
        --img_width 64 \
        --img_channel 1 \
        --input_length 10 \
        --total_length 20 \
        --num_hidden 128,128,128,128 \
        --filter_size 5 \
        --stride 1 \
        --patch_size 4 \
        --layer_norm 0 \
        --decouple_beta 0.1 \
        --reverse_scheduled_sampling 1 \
        --r_sampling_step_1 25000 \
        --r_sampling_step_2 50000 \
        --r_exp_alpha 2500 \
        --lr 0.0001 \
        --batch_size 8 \
        --max_iterations 30000 \
        --display_interval 100 \
        --test_interval 100000 \
        --snapshot_interval 500 \
        --pretrained_model ./training_results/mnist_predrnn_v2_baseline/model.ckpt-$n

    mv ./tmp_test_results.txt ./test_results/mnist_predrnn_v2_baseline/$n
done