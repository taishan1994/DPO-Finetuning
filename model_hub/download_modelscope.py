from modelscope.hub.snapshot_download import snapshot_download

model_name = "qwen/Qwen2-0.5B-Instruct"
model_dir = snapshot_download(model_name, cache_dir='./')

