from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from utility import load_texts

sys.stdout.reconfigure(encoding="utf-8")

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# # Set pat_token_id
# # tokenizer.pad_token = tokenizer.eos_token

# # prompt = "I'll be there for you when your wings break, letting go of you that's a heartbreak. It's buttferfly effect zone - I'm falling into your kaleidoscope! Is Daisy male or female?"
# # prompt = "I'll be there for you when your wings break, letting go of you that's a heartbreak. It's buttferfly effect zone - I'm falling into your kaleidoscope! Why I'll be there?"
# prompt = '"She likes coming here because she has fallen badly in love with you and Garth," he said.  "But she’s a good sort, and I’m grateful for all she has taught me.  Do you know, I think she tries to imitate you.  She looks cleaner, somehow—and I’ll swear she brushes that queer short mop of hers more than she used."'

# tokens = tokenizer.tokenize(prompt)
# print(tokens)

# # tokenize the text with padding and attention mask:
# inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
# for k, v in inputs.items():
# 	print(k, ":", v)
# print(tokenizer.decode(inputs.input_ids[0]))

# # print(tokenizer(prompt, return_tensors="pt"))

# # Outputs:
# outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=100)
# print(outputs[0])
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("Generated text:", generated_text)


# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)

# gen_text = tokenizer.batch_decode(gen_tokens)[0]


file_path = "./data-tr/holmes/GMNTL10.TXT"
# stories = list(load_texts("./data-tr/holmes/", keep_stop_word=True, tokenizer="gpt2"))
with open(file_path, 'r', encoding='latin-1') as fin:
	skip = True
	text = ""
	for line in fin:
		if not skip and not "project gutenberg" in line.lower():
			text += line
		elif "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS" in line:
			skip = False

tokens = tokenizer.tokenize(text)
print(len(tokens))
max_length = 100000000  # Set the maximum length for your model
encoded_input = tokenizer.encode_plus(
    text,
    max_length=max_length,
    truncation=False,
    return_tensors='pt',  # Return PyTorch tensors
    return_attention_mask=True
)
input_ids = encoded_input['input_ids']
print(len(input_ids[0]))
tokens_new = tokenizer.convert_ids_to_tokens(input_ids[0])
print(len(tokens_new))
print(len(set(tokens_new)))

