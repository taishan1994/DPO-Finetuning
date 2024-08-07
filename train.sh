CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file accelerate_config/default_config.yaml \
finetune_dpo.py \
train_config/qwen2_dpo_config.yaml \
--output_dir=output/qwen2-0.5B-Instruct-DPO \
--learning_rate=1e-5 \
--logging_dir=logs \
--model_name_or_path=./model_hub/Qwen2-0.5B-Instruct \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps=4 \
--save_steps=100 \
--max_prompt_length=128 \
--max_length=1024 \
--beta=0.1 \
--gamma=0.1