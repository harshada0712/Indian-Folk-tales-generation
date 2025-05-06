from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_model(model_path):
    """Load fine-tuned model and tokenizer"""
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

def generate_story(prompt, model, tokenizer, device, max_length=200, temperature=0.8):
    """Generate story from prompt"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)