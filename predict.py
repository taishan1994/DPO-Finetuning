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