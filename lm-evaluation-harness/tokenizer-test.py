from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gt2")

prompt = "I'll be there for you when your wings break, letting go of you that's a heartbreak."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)

gen_text = tokenizer.batch_decode(gen_tokens)[0]

