import json
import codecs
import sys

json_file = "../tokens-saved/babylm-10M-gutenberg-gpt2.json"
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

with open(json_file, 'r', encoding='utf-8') as file:
	data = json.load(file)

print(len(data))
trouble_tokens_1 = []
trouble_tokens_2 = []
after_correction_1 = []
after_correction_2 = []
count = 0
for token in data:
	# if '"' or '\\' in token:
	# 	if "\\u0120" not in token:
	# 		trouble_tokens.append(token)
	# if '"' in token:
	# 	trouble_tokens_1.append(token)
	# 	token_new = token.replace('"', "''")
	# 	after_correction_1.append(token_new)

	# if "\\\\" in token:
	# 	print(token)
	# 	trouble_tokens_2.append(token)
	# 	token_new = token.replace('\\', '\\\\')
	# 	after_correction_2.append(token_new)

	if '],"' in token:
		print(token)


print(set(trouble_tokens_1))
print(set(trouble_tokens_2))
print(set(after_correction_1))
print(set(after_correction_2))


"""
{'":', ',"', '"(', '")', '_"', '\'."', '\'"',
 '_("', '.,"', '"\'', ']."', '\',"', '___"', '").',
  '.\'"', '"?', '".', '."[', '?".', ':"', '_".', '"]',
   '?\'"', '"...', ')."', ';"', ',\'"', '!\'"', '";', '"),',
    '_"_', '_"\'', '_"[', '."', '"', '",', '],"', '_",', '..."',
     '_"...', ')"', '!"', '?"', '),"', '"___'}
"""