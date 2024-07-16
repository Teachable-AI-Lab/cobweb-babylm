import os
import json
import random as rd
from random import random, shuffle
from collections import Counter
from tqdm import tqdm
import argparse
import math
from multiprocessing import Pool
import pandas as pd
import datetime

from utility import process_text, load_texts, process_file, story2instances, question2instance, lemmatized_options
from cobweb.cobweb import CobwebTree
from cobweb.visualize import visualize

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--type", type=str, default="100M", choices=["10M", "100M", "dev", "test", "holmes"],
		help="The type of data used")
	parser.add_argument("--corpus", type=str, default="all", 
		choices=["all", "bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"],
		help="The corpus used")
	parser.add_argument("--tokenizer", type=str, default="spacy", choices=["gpt2", "spacy"],
		help="The tokenizer used")
	parser.add_argument("--scheme", type=str, default="inverse", choices=["inverse", "linear", "exp", "gaussian"], 
		help="The scheme for assigning weights to context words when generating an instance")
	parser.add_argument("--alpha", type=float, default=1.0, help="The alpha value when scheme == exp")
	parser.add_argument("--sigma", type=float, default=1.0, help="The sigma value when scheme == gaussian")
	# number of instances when least_freq = 1 for each data separately:
	# [46822, 337768, 1001630, 779349, 734587, 982011] (total 3882167)

	# Other configurations did before:
	parser.add_argument("--window", type=int, default=10, help="The window size (should be int greater than 0)")
	parser.add_argument("--least-freq", type=int, default=1, help="The tokens having the least frequency in the instances")
	parser.add_argument("--seed", type=int, default=123, help="random seed")

	parser.add_argument("--index-start", default=None, help="the index where the training instance starts")
	parser.add_argument("--index-end", default=None, help="the index where the training instance ends")
	parser.add_argument("--n-tr-splits", type=int, default=1, help="number of training splits (checkpoints)")
	parser.add_argument("--test", action="store_true", help="whether set up a test portion of data")
	parser.add_argument("--tr-portion", type=float, default=1., help="the portion of data used for training")
	parser.add_argument("--tr-split-size", type=int, help="the tr split size")
	parser.add_argument("--bidirection", action="store_true", help="unidirectional or bidirectional")

	parser.add_argument("--start-split", type=int, default=0, help="the split index to start with when training")
	parser.add_argument("--load-model", action="store_true", help="whether to load the existing model file")
	parser.add_argument("--load-model-file", type=str, help="the model path being loaded")

	parser.add_argument("--show-vocab", action="store_true", help="whether to display the vocab size and store as filename for each training split")
	parser.add_argument("--token-form", type=str, default="str", choices=["str", "encoding"],
	 	help="whether to store the tokens as encodings instead of real string tokens")
	args = parser.parse_args()

	# Preprocess configuration:
	data_type = args.type
	data_corpus = args.corpus
	tokenizer_choice = args.tokenizer
	scheme = args.scheme
	alpha_exp = args.alpha
	sigma_gaussian = args.sigma
	token_form = args.token_form

	story_path = "./data-tr/"
	if data_type in ["10M", "100M"]:
		story_path += f"train_{data_type}/"
	else:
		story_path += f"{data_type}/"
	if data_corpus != "all":
		story_path += f"{data_corpus}.train" if data_type in ["10M", "100M"] else f"{data_corpus}.{data_type}"
	json_name = f"./tokens-saved/babylm-{data_type}-{data_corpus}-{tokenizer_choice}-{token_form}.json"

	verbose = True
	limit = None
	window = args.window
	least_frequency = args.least_freq
	seed = args.seed
	# window = 10  # the size of the "gram" (so context words = 2 * window)
	# least_frequency = 1  # used to filter out the words having frequency less than some specified one.
	# seed = 123  # random seed for shuffling instances
	keep_stop_word = True
	stop_word_as_anchor = True
	preview = True  # to preview the generated tokens and preprocessed instances

	# Trained instances configuration:
	# First specify the number of instances considered. If None, then all generated instances will be considered
	# start_instance_id = None 
	# end_instance_id = None
	start_instance_id = args.index_start
	end_instance_id = args.index_end
	# Then specify the size of training set by specifying the training portion
	# train_portion = 1  # should always be 1 since there is a separate dev and test set
	train_portion = args.tr_portion
	have_test_portion = True if args.test else False  # should always be False since there is a separate dev and test set
	# Finaly specify the number of training splits generated in the training set
	# Priority order: n_tr_splits > split_size
	# n_tr_splits = 1
	# split_size = 833333
	n_tr_splits = args.n_tr_splits
	split_size = args.tr_split_size
	bidirection = True if args.bidirection else False  # generate instances with context words just before the anchor, or both before and after the anchor
	train_date = datetime.datetime.now().strftime("%m%d%Y")

	# Training process configuration:
	start_split = 0
	load_model = False
	# load_model_file = f"./model/cobweb-seed123-window{window}-leastfreq{least_frequency}-instances416667-split{start_split-1}.json"
	# load_model_file = "./model-saved/cobweb-seed123-window10-leastfreq3-instances5000000-split0.json"
	start_split = args.start_split
	load_model = True if args.load_model else False
	load_model_file = args.load_model_file


	""" Load and preprocess the text data used """
	if verbose:
		print("Start the loading and preprocessing process.")

	# Create the integrated text from several text file (stories):
	if not os.path.isfile(json_name):
		if verbose:
			print("\nReading and preprocessing...")
		# if dummy:
		# 	stories_dir = "./data-tr-dummy"
		# else:
		# 	stories_dir = "./data-tr"
		if data_corpus == "all":
			stories = list(load_texts(story_path, limit=limit, keep_stop_word=keep_stop_word, 
				tokenizer=tokenizer_choice, token_form=token_form))
		else:
			stories = process_file(story_path, keep_stop_word=keep_stop_word, 
				tokenizer=tokenizer_choice, token_form=token_form)
		with open(json_name, "w") as fout:
			json.dump(stories, fout, indent=4)
	else:
		if verbose:
			print("\nLoading the preprocessed stories...")
		with open(json_name, "r") as fin:
			stories = json.load(fin)

	if data_corpus != "all":
		stories = [stories]

	overall_freq = Counter([word for story in stories for word in story])
	if preview:
		# A 200-first-word preview of some preprocessed story:
		print("\nPreview of the 200 first words of the first preprocessed story:")
		print(stories[0][:200])
		# You may see the overall word frequencies (50 most frequent words):
		print("\npreview of the 100 most frequent words frequency:")
		print(overall_freq.most_common(100))


	""" Generate the instances for Cobweb learning, the 'n-grams' """
	# Filter out the words having frequency >= least_frequency only:
	stories = [[word for word in story if overall_freq[word] >= least_frequency] for story in stories]
	vocab_size = len(Counter([word for story in stories for word in story]))
	print("\nVocab Size:", vocab_size)
	print("Counting seprately:", [len(Counter(story)) for story in stories])


	# Generate instances (along with their story and anchor indices):
	print("\nNow generate the instances:")
	instances = []
	with Pool() as pool:
		processed_stories = pool.starmap(
			story2instances, 
			[(story, window, stop_word_as_anchor, scheme, bidirection, alpha_exp, sigma_gaussian) for story in stories]
		)
		# print(len(stories))
		# print(len(processed_stories))
		print("The number of instances for each story:", [len(story_instances) for story_instances in processed_stories])
		# for story_idx, story_instances in enumerate(processed_stories):
		for story_instances in tqdm(processed_stories):
			for anchor_idx, instance in story_instances:
				# instances.append((instance, story_idx, anchor_idx))
				instances.append(instance)

	rd.seed(seed)
	shuffle(instances)

	print("\nThe number of instances: {}".format(len(instances)))  # 16645730 (when least_frequency=3 and window=10)
	if preview:
		print("The first 5 instances:")
		for i in range(5):
			print(instances[i])


	"""
	Generate training and test sets (within the Holmes data - not the external test set!)
	We generate n_tr_splits training splits + test split.
	After training all training splits, we store the model once.
	Then use it to test with the test split, and after that, train the additional test instances.
	Store the model another time.
	"""
	# if n_instances:
	# 	instances = instances[:n_instances]
	# if start_instance_id:
	# 	instances = instances[start_instance_id:]
	start_id = start_instance_id
	end_id =  end_instance_id
	instances = instances[start_id:end_id]
	n_instances = len(instances)
	tr_size = round(n_instances * train_portion)
	if n_tr_splits is not None:
		tr_split_size = round(tr_size / n_tr_splits)
	else:
		tr_split_size = split_size
		n_tr_splits = math.ceil(tr_size / tr_split_size)
	instances_splits = []
	for i in range(n_tr_splits):
		if i != n_tr_splits - 1:
			instances_splits.append(instances[i*tr_split_size:(i+1)*tr_split_size])
		else:
			instances_splits.append(instances[i*tr_split_size:tr_size])
	# instances_splits.append(instances[tr_size:])
	if have_test_portion:
		instances_te = instances[tr_size:]
		# print(instances_te[:5])
		instances_te_no_anchor = [{'context-before': instance['context-before'], 'context-after': instance['context-after']} for instance in instances_te]
	print(f"\n Have test set? {have_test_portion}")
	print(f"Here consider instance {start_instance_id} to instance {end_instance_id}.")
	print(f"There are {len(instances_splits)} training sets in total, and their sizes are {[len(split) for split in instances_splits]}.")
	if have_test_portion:
		print(f"There are {len(instances_te)} instances in the test set.")


	""" Model initialization """
	print(f"\nModel initialization: Load model - {load_model}.")
	tree = CobwebTree(0.000001, False, 0, True, False)
	if load_model:
		with open(load_model_file, 'r') as file:
			model_data = file.read()
		tree.load_json(model_data)


	""" Train Cobweb with the training splits """
	print("\nStart Training process.")

	if args.show_vocab:
		vocab = set()

	for i in range(start_split, len(instances_splits)):
		print(f"\nNow train split {i + 1} / {len(instances_splits)}.")
		print(f"The number of instances: {len(instances_splits[i])}")

		for instance in tqdm(instances_splits[i]):
			tree.ifit(instance)
			if args.show_vocab:
				vocab.update(instance['anchor'])
				if bidirection:
					vocab.update(instance['context-before'])
					vocab.update(instance['context-after'])
				else:
					vocab.update(instance['context'])
		if args.show_vocab:
			print(f"Vocab size of the current tree: {len(vocab)}")

		# After training, store the model (json file):
		json_output = tree.dump_json()
		file_name = f"./model-saved/cobweb/cobweb-data{data_type}-{data_corpus}-tokenizer_{tokenizer_choice}-token_form_{token_form}-scheme_{scheme}"
		file_name += f"-seed{seed}-window{window}-leastfreq{least_frequency}-instances{len(instances_splits[i])}-split{i}-{train_date}"
		if args.show_vocab:
			file_name += f"-vocab{len(vocab)}"
		file_name += ".json"
		print("Now save the model checkpoint:")
		with open(file_name, 'w', encoding="utf-8") as json_file:
			json_file.write(json_output)


	""" Test the Cobweb trained with all training splits """
	# print("\nNow all the train splits are used. Test with the test data.")
	# anchors_te = [list(instance['anchor'].keys())[0] for instance in instances_te]
	# n_correct = 0
	# for i in tqdm(range(len(instances_te_no_anchor))):
	# 	probs_pred = tree.predict_probs(instances_te_no_anchor[i], nodes_pred, False, False)
	# 	anchor_pred = sorted([(probs_pred["anchor"][word], random(), word) for word in probs_pred['anchor']], reverse=True)[0][2]
	# 	if anchor_pred == anchors_te[i]:
	# 		n_correct += 1
	# test_acc = n_correct / len(instances_te)
	# print(f"Accuracy on the test data within the raw data: {test_acc} ({n_correct}/{len(instances_te)}).")


	""" Train Cobweb with the additional test set """
	if have_test_portion:
		print("\nNow train with the additional test set.")
		for instance in tqdm(instances_te):
			tree.ifit(instance)
		# After training, store the model (json file):
		json_output = tree.dump_json()
		file_name = f"./model-saved/cobweb/cobweb-data{data_type}-{data_corpus}-seed{seed}-window{window}-leastfreq{least_frequency}-all-instances-{train_date}.json"
		with open(file_name, 'w') as json_file:
			json_file.write(json_output)

	print("\nTraining process is done!")

