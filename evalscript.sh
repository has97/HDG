task='img_dg'
is_data_aug=yes
loader_name='daml_loader'
algorithms=('SCIPD')
algorithm_num=${#algorithms[@]}

CUDA_VISIBLE_DEVICES=4 python eval_hscore.py --task $task --loader_name $loader_name --is_data_aug $is_data_aug