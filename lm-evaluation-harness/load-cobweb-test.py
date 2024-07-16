from cobweb.cobweb import CobwebTree
import json
import re

tree = CobwebTree(0.000001, False, 0, True, False)
model_file = "../model-saved/cobweb/cobweb-data10M-all-tokenizer_gpt2-scheme_inverse-seed123-window10-leastfreq1-instances1189512-split0-06232024-vocab43381.json"
# model_file = "../model-saved/cobweb/cobweb-data10M-gutenberg-tokenizer_gpt2-seed123-window10-leastfreq1-instances3551073-split0-06052024.json"
# model_file = "../../cobweb-language-exp/model/cobweb-seed123-window10-leastfreq3-instances416667-split10.json"
# model_file = "../model-saved/cobweb/cobweb-data10M-all-tokenizer_gpt2-scheme_inverse-seed123-window10-leastfreq1-instances1208188-split0-06132024-vocab43190.json"
with open(model_file, 'r') as file:
	model_data = file.read()
# # model_data = model_data.replace(':', '":"')
print("load json file succeeded")

num_lines = 10

processed_types = []

# Load the JSON file

# # Convert JSON data to string for line by line access
# data_str = json.dumps(model_data, indent=4)
# lines = data_str.splitlines()

# # Display the first few lines
# print("First {} lines:".format(num_lines))
# for line in lines[:num_lines]:
#     print(line)

# # Display the last few lines
# print("\nLast {} lines:".format(num_lines))
# for line in lines[-num_lines:]:
#     print(line)

# with open(model_file, 'r') as file:
#    	data_dict = json.load(file)

# def fix_json(json_content):
#     try:
#         # Try to load the JSON content directly
#         return json.loads(json_content)
#     except json.JSONDecodeError as e:
#         # If there's an error, print the error message
#         print(f"JSON decoding error: {e}")
        
#         # Fix the specific error (example provided for ":" value case)
#         # This part should be customized based on your specific JSON issues
#         fixed_content = json_content.replace(':', '":"').replace('""', '":"').replace(',', '","')
        
#         try:
#             # Try to load the fixed content
#             return json.loads(fixed_content)
#         except json.JSONDecodeError as e:
#             print(f"JSON decoding error after fix: {e}")
#             return None

def print_error_context(json_content, error, context_radius=50):
    error_pos = error.pos
    line_id = json_content.count('\n', 0, error_pos) + 1
    start = max(0, error_pos - context_radius)
    end = min(len(json_content), error_pos + context_radius)
    # print(f"Error at position ({line_id}, {error_pos}):")
    # print_context(json_content. start, end)
    error_context = json_content[start:end]
    print(f"Error at position ({line_id}, {error_pos}):")
    print(f"Context around the error:\n{error_context}")
    return error_pos


# def print_ori_context(json_content, line_id, pos, context_radius=50):
# 	context = json_content[start:end]
# 	print(f"Context around the position:\n{error_context}")


def load_json_check(json_content):
	# Logic: Substitute the double quotes in the keys as single quotes
	# Better change the whole key (with double quotes) so other keys won't be affected
	# json_content = re.sub(r'""\,""', r'"\",\""')  # 3253860473632812, "","": 0.9083333015441
	# json_content = json_content.replace('"",""', '\"\",\"\"')
	json_content = json_content.replace('"","": ', '"\',\'": ')  # 3253860473632812, "","": 0.9083333015441
	json_content = json_content.replace('"["": ', '"[\'": ')  # 968124389648438, "["": 12.60317611694335
	json_content = json_content.replace('"\',"": ', '"\',\'": ')  # 44448947906494, "',"": 3.348412990570068
	json_content = json_content.replace('"\'"": ', '"\'\': ')  # 849220275878906, "'"": 6.713095188140869
	json_content = json_content.replace('""": ', '"\'": ')  # 3849220275878906, """: 6.713095188140869
	json_content = json_content.replace('"\'\': ', '"\'": ')  # "'': 6.71309518814086914
	json_content = json_content.replace('":"": ', '":\'": ')  # ":"": 3.60238122940063477
	return json_content


def preprocess_line(line):
	# Split the line into key-value pairs
	pairs = line.split(', ')
	
	processed_pairs = []
	for pair in pairs:
	    if ':' in pair:
	        # Find the position of the last colon
	        last_colon_pos = pair.rfind(':')
	        first_colon_pos = pair.find('"')
	        
	        # Extract the key
	        key = pair[first_colon_pos+1:last_colon_pos-1]
	        
	        # Substitute all double quotes inside the key with single quotes
	        # if ": {" not in key and '"' in key:
	        # 	processed_types.append(key)
	        # 	key = key.replace('"', "'")

	        if '\\' in key:
	        	processed_types.append(key)
	        	key = key.replace('\\', '\\\\')
	        
	        # Reconstruct the key-value pair
	        value = pair[last_colon_pos+1:].strip()
	        processed_pair = pair[:first_colon_pos] + f'"{key}": {value}'
	        processed_pairs.append(processed_pair)
	    else:
	        processed_pairs.append(pair)
    
	# Join the processed key-value pairs back into a single line
	processed_line = ', '.join(processed_pairs)
    
	return processed_line


def preprocess_all_lines(json_content):
	for line in json_content:
		line = preprocess_line(line)
	return json_content


def transform_inner_quotes_in_keys(json_content):
    # Regular expression to find keys wrapped in double quotes
    def replace_inner_quotes(match):
        key = match.group(1)
        value = match.group(2)
        # Replace double quotes inside the key with single quotes
        transformed_key = key.replace('"', "'")
        # Return the key wrapped in double quotes
        return f'"{transformed_key}": {value}'

    # Use regex to find all keys in the JSON content
    # fixed_content = re.sub(r'"([^"]*?)"\s*:', replace_inner_quotes, json_content
    # fixed_content = re.sub(r'"([^"]*?)":', replace_inner_quotes, json_content)
    # fixed_content = re.sub(r'"([^"]*)"\s*:\s*([^,{}]*)', replace_inner_quotes, json_content)
    return fixed_content


def load_json_with_context(filepath, context_radius=50):
    with open(filepath, 'r') as file:
        json_content = file.read()
        # json_content = json_content.replace('","', ',')
        json_content = load_json_check(json_content)
        # json_content = transform_inner_quotes_in_keys(json_content)
    
    try:
        data = json.loads(json_content)
        return data
    except json.JSONDecodeError as e:
        print_error_context(json_content, e, context_radius)
        return None
    # data = json.loads(json_content)



# output_file = '../model-saved/cobweb/cobweb-data10M-all-tokenizer_gpt2-scheme_inverse-seed123-window10-leastfreq1-instances1208188-split0-06132024-vocab43190-preprocessed.json'
output_file = "../model-saved/cobweb/"
output_file += "cobweb-data10M-all-tokenizer_gpt2-scheme_inverse-seed123-window10-leastfreq1-instances1189512-split0-06232024-vocab43381.json"
def load_json_with_context_new(filepath, context_radius=150, load=False):
	if not load:
		# Preprocess and overwrite:
		with open(filepath, 'r') as infile, open(output_file, 'w') as outfile:
		    for line in infile:
		        processed_line = preprocess_line(line.strip())
		        outfile.write(processed_line + '\n')
	with open(output_file, 'r') as outfile:
		json_content = outfile.read()
	try:
	    data = json.loads(json_content)
	    return data
	except json.JSONDecodeError as e:
	    error_pos = print_error_context(json_content, e, context_radius)
	    print("\n=======\nFYI here is the context of original file:")
	    with open(filepath, 'r') as infile:
	    	json_content_ori = infile.read()
	    start = max(0, error_pos - context_radius)
	    end = min(len(json_content_ori), error_pos + context_radius)
	    print(json_content_ori[start:end])
	    return None


# data_dict = json.loads(model_data)
# data_dict = fix_json(model_data)
# if data_dict:
# # print(data_dict.keys())
# 	print(len(data_dict))
# else:
# 	print("Failed to parse.")
# print(len(data_dict))


# data = load_json_with_context_new(
# 		"../model-saved/cobweb/cobweb-data10M-all-tokenizer_gpt2-scheme_inverse-seed123-window10-leastfreq1-instances1189512-split0-06232024-vocab43381.json",
# 		context_radius=150,
# 		load=True,
# 	)
# if data:
# 	print(len(data))
# else:
# 	print("Failed to load JSON")

# print(set(processed_types))

tree.load_json(model_data)
print("load json file into CobwebTree succeeded")

# dummy_inst = {"anchor": {}, "context": {}}
# probs_pred = tree.predict_probs(dummy_inst, 100, False, False)
# anchors = list(probs_pred['anchor'].keys())
# contexts_before = list(probs_pred['context-before'].keys())
# contexts_after = list(probs_pred['context-after'].keys())
# print(len(anchors))
# print(len(contexts_before), len(contexts_after))
# vocab = set(anchors + contexts_before + contexts_after)
# print(len(vocab))

