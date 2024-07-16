import re
import os
import sys
import spacy
from multiprocessing import Pool
import numpy as np
from spacy.tokenizer import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import preprocess_corpus

# Import the sentence processor from spacy
nlp = spacy.load("en_core_web_sm", disable = ['parser'])
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')

# Import the tokenizer from GPT-2 (Hugging Face):
sys.stdout.reconfigure(encoding="utf-8")
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

# Nishant's processor:
# nlp = spacy.blank("en")
# nlp.max_length = float('inf')
# infixes = tuple([r"\w+'s\b", r"\w+'t\b", r"\d+,\d+\b", r"(?<!\d)\.(?!\d)"] +  nlp.Defaults.prefixes)
# infix_re = spacy.util.compile_infix_regex(infixes)
# nlp.tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

# Test:
# tokens = nlp("I ate Jack's lunch this morning, and it costed me 1,000 dollars - Jeez! I can't make it! 15.5?")
# for token in tokens:
# 	print(token)
# text = [token.lemma_.lower() for token in tokens if (not token.is_punct)]
# print(text)
# text = [token.text.lower() for token in tokens if (not token.is_punct)]
# print(text)
# text = [token.lemma_.lower() for token in tokens if (not token.is_punct and not token.is_stop)]
# print(text)


""" The following functions are used for preprocessing in loading the stories """

def process_text(text, keep_stop_word=False, preview=True, tokenizer="spacy", token_form="str"):
	if tokenizer == "spacy":
		punc = re.compile(r"[^_a-zA-Z0-9,.!?:;'\s]")
		whitespace = re.compile(r"\s+")
		text = re.sub(r'--', r' ', text)
		text = punc.sub("", text)
		text = whitespace.sub(" ", text)
		text = text.strip().lower()
		# parse:
		print("\nThe text before preprocess (the first 200 words):")
		print(' '.join(text.split()[:200]))
		text = nlp(text)
		if keep_stop_word:
			text = [token.lemma_.lower() for token in text if (not token.is_punct)]
		else:
			text = [token.lemma_.lower() for token in text if (not token.is_punct and not token.is_stop)]
	elif tokenizer == "gpt2":
		# To avoid the troubles one may have when training cobweb tree and include punctuations as keys in dictionaries,
		# Make a uniform convention towards the puncutation:
		# convention = {
		# 	'"': "'",
		# 	'\\': '\\\\',
		# }
		# for text_bf, text_af in convention.items():
		# 	text = text.replace(text_bf, text_af)
		if token_form == "str":
			text = tokenizer_gpt2.tokenize(text)
			text_new = []
			for token in text:
				# if '"' or '\\' in token:
				# 	print()
				token_new = token.replace('"', "''").replace("\\", "\\\\")
				text_new.append(token_new)
		else:
			# Store tokens as encodings.
			# First detect if the length is greater than the max length
			text_new = [str(index) for index in tokenizer_gpt2.encode(text)]
			# and in this way, we won't have to encounter any punctuation issue
	return text_new


def process_file(file, keep_stop_word=True, tokenizer="spacy", token_form="str"):
	# first ensure the file is the one we need.
	# if not re.search(r'.*\.(train|dev|test)$', os.path.basename(file)):
	# 	return None
	# print(os.path.basename(file))
	# print(bool(re.match(r'.*\.(train|dev|test|TXT|txt)$', os.path.basename(file))))
	if not bool(re.match(r'.*\.(train|dev|test|TXT|txt)$', os.path.basename(file))):
		return None
	print(f"Processing file: {file}")

	# Then find the corpus used by the file name:
	dot_index = os.path.basename(file).rfind('.')
	# print(os.path.basename(file)[:dot_index])
	if dot_index != -1 and dot_index != 0:
		data_corpus = os.path.basename(file)[:dot_index]
	else:
		return None
	# print(data_corpus)
	with open(file, 'r', encoding='utf-8') as fin:
		if data_corpus == "bnc_spoken":
			text = ""
			for line in fin:
				text += line
		elif data_corpus == "childes":
			text = preprocess_corpus.text_childes(fin)
		elif data_corpus == "gutenberg":
			text, n_novels_gutenberg = preprocess_corpus.text_gutenberg(fin, return_n_novels=True)
			print(f"\nThere are {n_novels_gutenberg} Gutenberg stories detected.")
		elif data_corpus == "open_subtitles":
			text = preprocess_corpus.text_open_subtitles(fin)
		elif data_corpus == "simple_wiki":
			text = preprocess_corpus.text_simple_wiki(fin)
		elif data_corpus == "switchboard":
			text = preprocess_corpus.text_switchboard(fin)
		else:
			# Load Holmes Stories:
			skip = True
			text = ""
			for line in fin:
				if not skip and not "project gutenberg" in line.lower():
					text += line
				elif "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS" in line:
					skip = False
		output = process_text(text, keep_stop_word=keep_stop_word, 
			tokenizer=tokenizer, token_form=token_form)
		return output


# def process_file(idx, name, fp, verbose=True, keep_stop_word=False):
# 	""" Load and preprocess a text file """
# 	if verbose:
# 		print("Processing file {} - {}".format(idx, name))
# 	if not re.search(r'.*\.(train|dev|test)$', name):
# 		return None
# 	with open(fp, 'r', encoding='latin-1') as fin:
# 		skip = True
# 		text = ""
# 		for line in fin:
# 			if not skip and not "project gutenberg" in line.lower():
# 				# In the considered text lines, disregard the ones with "project gutenberg"
# 				text += line
# 			elif "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS" in line:
# 				# The text before "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS" should not be considered in the text
# 				skip = False

# 		output = process_text(text, keep_stop_word=keep_stop_word)
# 		return output


def load_texts(training_dir, limit=None, keep_stop_word=False, tokenizer="spacy", token_form="str"):
	""" Load the preprocessed texts used for training """
	for path, subdirs, files in os.walk(training_dir):
		if limit is None:
			limit = len(files)
		texts = [(os.path.join(path, name), keep_stop_word, tokenizer) for idx, name in enumerate(files[:limit])]

		# Preprocess the text files in parallel
		if tokenizer == "spacy":
			with Pool() as pool:
				outputs = pool.starmap(process_file, texts)
				for output in outputs:
					if output is None:
						continue
					yield output
		else:
			# Hugging Face does not support parallelization
			for path, keep_stop_word, tokenizer in texts:
				yield process_file(path, keep_stop_word, tokenizer, token_form)




""" The following functions are used for the instances (n-grams) generation: anchor + context words """

def old_get_instance(text, anchor_idx, anchor_wd, window):
	""" Generate an instance {'anchor': {anchor_word: 1}, 'context': {context_1: ..., context_2: ..., ...}} """
	before_text = text[max(0, anchor_idx - window):anchor_idx]
	after_text = text[anchor_idx + 1:anchor_idx + 1 + window]
	ctx_text = before_text + after_text
	ctx = {}

	# In a language task, the context words are not considered as simple counts.
	# Considering the proximity to the anchor word, the further the context word to the anchor, the less weight it will have
	for i, w in enumerate(before_text):
		ctx[w] = 1 / abs(len(before_text) - i)
	for i, w in enumerate(after_text):
		ctx[w] = 1 / (i + 1)

	instance = {}
	instance['context'] = ctx
	if anchor_wd is None:
		return instance
	instance['anchor'] = {anchor_wd: 1}
	return instance


def get_instance(text, anchor_idx, anchor_wd, window, scheme="inverse", bidirection=False, alpha=1.0, sigma=1.0):
	""" Generate an instance {'anchor': {anchor_word: 1}, 'context': {context_1: ..., context_2: ..., ...}} """
	""" Introduce context words before and after the anchor word."""

	instance = {}
	if bidirection:
		before_text = text[max(0, anchor_idx - window):anchor_idx]
		after_text = text[anchor_idx + 1:min(anchor_idx + 1 + window, len(text))]
		context_before = {}
		context_after = {}
		# In a language task, the context words are not considered as simple counts.
		# Considering the proximity to the anchor word, the further the context word to the anchor, the less weight it will have
		for i, w in enumerate(before_text):
			distance = len(before_text) - i
			context_before[w] = compute_weight(distance=distance, scheme=scheme, alpha=alpha, sigma=sigma)
			# context_before[w] = 1 / abs(len(before_text) - i)
		for i, w in enumerate(after_text):
			distance = i + 1
			# context_after[w] = 1 / (i + 1)
			context_after[w] = compute_weight(distance=distance, scheme=scheme, alpha=alpha, sigma=sigma)

		instance['context-before'] = context_before
		instance['context-after'] = context_after

	else:
		before_text = text[max(0, anchor_idx - window):anchor_idx]
		context = {}
		for i, w in enumerate(before_text):
			distance = len(before_text) - i
			context[w] = compute_weight(distance=distance, scheme=scheme, alpha=alpha, sigma=sigma)
			# context[w] = 1 / abs(len(before_text) - i)
		instance['context'] = context

	if anchor_wd is None:
		return instance
	instance['anchor'] = {anchor_wd: 1}
	return instance


def question2instance(text, window, test=False, keep_stop_word=False):
	""" Load and preprocess a single (line) of text """
	# Preprocess
	instance = {}
	if test:
		# punc = re.compile(r"[^_a-zA-Z,.!?:;\s]")
		punc = re.compile(r"[^_a-zA-Z0-9,.!?:;'\s]")
	else:
		# punc = re.compile(r"[^a-zA-Z,.!?:;\s]")
		punc = re.compile(r"[^a-zA-Z0-9,.!?:;'\s]")
	whitespace = re.compile(r"\s+")
	text = punc.sub("", text)
	text = whitespace.sub(" ", text)
	text = text.strip().lower()
	
	# Parse
	text = nlp(text)
	anchor_id = 0
	# find the anchor id
	for i in range(len(text)):
		if "_" in text[i].text:
			anchor_id = i
			break
	if keep_stop_word:
		before_anchor = [token.lemma_.lower() for token in text[max(0, anchor_id - window):anchor_id] if (not token.is_punct)]
		after_anchor = [token.lemma_.lower() for token in text[anchor_id + 1:min(len(text), anchor_id + 1 + window)] if (not token.is_punct)]
	else:
		before_anchor = [token.lemma_.lower() for token in text[max(0, anchor_id - window):anchor_id] if (not token.is_punct and not token.is_stop)]
		after_anchor = [token.lemma_.lower() for token in text[anchor_id + 1:min(len(text), anchor_id + 1 + window)] if (not token.is_punct and not token.is_stop)]
	context_before = {}
	context_after = {}
	for i, w in enumerate(before_anchor):
		context_before[w] = 1 / abs(len(before_anchor) - i)
	for i, w in enumerate(after_anchor):
		context_after[w] = 1 / (i + 1)
	instance['context-before'] = context_before
	instance['context-after'] = context_after
	return instance


def lemmatized_options(option_instance):
	instance = {}
	for k, v in option_instance.items():
		v = nlp(v)
		instance[k] = [token.lemma_.lower() for token in v][0]
	return instance


def _story2instances(story, window, stop_word_as_anchor=False, scheme="inverse", bidirection=False, alpha=1.0, sigma=1.0):
	for anchor_idx, anchor_wd in enumerate(story):
		if stop_word_as_anchor:
			yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window=window, scheme=scheme, bidirection=bidirection, alpha=alpha, sigma=sigma)
		else:
			if anchor_wd not in nlp.Defaults.stop_words:
				yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window=window, scheme=scheme, bidirection=bidirection, alpha=alpha, sigma=sigma)


def story2instances(story, window, stop_word_as_anchor=False, scheme="inverse", bidirection=False, alpha=1.0, sigma=1.0):
	return list(_story2instances(story, window, stop_word_as_anchor=stop_word_as_anchor, scheme=scheme, bidirection=bidirection, alpha=alpha, sigma=sigma))


def compute_weight(distance, scheme="inverse", alpha=1.0, sigma=1.0):
	"""
	The function to calculate the corresponding weight for a context word
	given its proximity to the anchor word and the chosen function.
	"""
	def weight_inverse(d):
		return 1 / abs(d)

	def weight_linear(d):
		return 1 / (abs(d) + 1)

	def weight_exp(d):
		return np.exp(-alpha * abs(d))

	def weight_gaussian(d):
		return np.exp(-(d ** 2) / (2 * sigma ** 2))

	weight_funcs = {
	"inverse": weight_inverse,
	"linear": weight_linear,
	"exp": weight_exp,
	"gaussian": weight_gaussian
	}
	weight_func = weight_funcs.get(scheme, weight_inverse)
	return weight_func(distance)







