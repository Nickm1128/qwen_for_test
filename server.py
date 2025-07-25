from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer  
import torch

app = Flask(__name__)

model_id = "/workspace/models/Qwen3-8B-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",             
    torch_dtype=torch.float16,     
    trust_remote_code=True         
)

@app.route('/chat', methods=['POST'])
def chat_completion():
    data = request.get_json(force=True)
    messages = data.get('messages')
    if not messages:
        return jsonify({'error': 'messages field missing'}), 400

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  
    )
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parse thinking content if present
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return jsonify({'thinking_content': thinking_content, 'content': content})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
