# DPO-Finetuning
专门用于训练DPO模型的仓库。

---

数据地址：链接: https://pan.baidu.com/s/1L01fhb40jJprlCmRKVq2ig?pwd=aspy 提取码: aspy

# 一般步骤

1. 将test.jsonl和train.jsonl下载到data/CValues-Comparison/下。

2. 运行model_hub/下的download_modelscope.py下载预训练的权重。

3. 训练`sh train.sh`

4. 预测：`python predict.py
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
   
   
   base_model_path = "./model_hub/Qwen2-0.5B-Instruct/"
   dpo_model_path = "./output/qwen2-0.5B-Instruct-DPO"
   
   
   device = "cuda"
   quantization_config = None
   model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                device_map="auto",
                                                trust_remote_code=True,
                                                quantization_config=quantization_config)
   tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
   model.eval()
   
   dpo_model = AutoModelForCausalLM.from_pretrained(dpo_model_path,
                                                device_map="auto",
                                                trust_remote_code=True,
                                                quantization_config=quantization_config)
   dpo_model.eval()
   
   
   
   def get_result(model_inputs, model):
       generated_ids = model.generate(
           **model_inputs,
           max_new_tokens=512,
           eos_token_id=tokenizer.get_vocab()["<|im_end|>"]
       )
       generated_ids = [
           output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
       ]
   
       response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
       return response
   
   while True:
       prompt = input(">>>")
   
       messages = [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": prompt}
       ]
       text = tokenizer.apply_chat_template(
           messages,
           tokenize=False,
           add_generation_prompt=True
       )
   
       # print(text)
   
       model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
       base_model_response = get_result(model_inputs, model)
       model_inputs = tokenizer([text], return_tensors="pt").to(dpo_model.device)
       dpo_model_response = get_result(model_inputs, dpo_model)
       print("基座模型：", base_model_response)
       print("DPO模型：", dpo_model_response)
       print("="*100)
   ```

# 如何训练其它模型

主要还是finetune_dpo.py里面的：apply_chat_template

```python
def apply_chat_template(example,
                        system="You are a helpful assistant."):
    # print(example)
    messages = example["messages"]
    chosen_message = ""
    rejected_message = ""
    chosen_score = 0.99
    rejected_score = 0.01

    # DPOTrainer里面会加一个Bos
    # prompt = "<|im_start|>system\n{}<|im_end|>\n".format(system)
    prompt = "system\n{}<|im_end|>\n".format(system)
    for i, message in enumerate(messages):
        role = message["role"]
        # print(message)
        if role == "user":
            value = message["value"]
            _input = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(value)
            prompt += _input
        else:
            chosen_value = message["chosen_value"]
            rejected_value = message["rejected_value"]
            if i != len(messages) - 1:
                # 如果是多轮对话， 前面几轮对话的choen和rejected应该是一样的
                prompt += chosen_value  + "<|im_end|>\n"
            else:
                # 最后面不需要再加一个<|im_end|>，DPOTrainer里面会加一个Eos
                chosen_message = chosen_value
                rejected_message += rejected_value
                chosen_score = message["chosen_score"]
                rejected_score = message["rejected_score"]

    example["prompt"] = prompt
    example["chosen"] = chosen_message
    example["rejected"] = rejected_message
    example["reference_chosen_logps"] = chosen_score
    example["reference_rejected_logps"] = rejected_score
    return example
```

需要转换为不同模型对应的输入的格式。另外需要提供对chosen和rejected的评分。

# 参考

> https://github.com/huggingface/alignment-handbook



