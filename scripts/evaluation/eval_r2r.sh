# set mp3d path
# export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

# set java path
# export JAVA_HOME=$java_path
# export PATH=$JAVA_HOME/bin:$PATH
# export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# activate environment
# conda activate navillm

torchrun --nnodes=1 --nproc_per_node=1 --master_port 41000 train.py \
    --stage multi --mode test --data_dir data --cfg_file configs/multi.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B --precision amp_bf16 \
    --resume_from_checkpoint data/model_with_pretrain.pt \
    --test_datasets R2R \
    --jsonpath ../../tasks/R2R/tmp/navillm_train_enc.json \
    --batch_size 4 --output_dir build/eval --validation_split train --save_pred_results

torchrun --nnodes=1 --nproc_per_node=1 --master_port 41000 train.py \
    --stage multi --mode test --data_dir data --cfg_file configs/multi.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B --precision amp_bf16 \
    --resume_from_checkpoint data/model_with_pretrain.pt \
    --test_datasets R2R \
    --jsonpath ../../tasks/R2R/tmp/navillm_val_seen_enc.json \
    --batch_size 4 --output_dir build/eval --validation_split val_seen --save_pred_results

torchrun --nnodes=1 --nproc_per_node=1 --master_port 41000 train.py \
    --stage multi --mode test --data_dir data --cfg_file configs/multi.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B --precision amp_bf16 \
    --resume_from_checkpoint data/model_with_pretrain.pt \
    --test_datasets R2R \
    --jsonpath ../../tasks/R2R/tmp/navillm_val_unseen_enc.json \
    --batch_size 4 --output_dir build/eval --validation_split val_unseen --save_pred_results