import os
import json
from collections import Counter
import argparse
import re
import spacy

# from utility import process_text, load_texts, story2instances, question2instance, lemmatized_options
from cobweb.cobweb import CobwebTree
from cobweb.visualize import visualize
import preprocess_corpus

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="100M", choices=["10M", "100M", "dev", "test"])
parser.add_argument("--corpus", type=str, default="bnc_spoken", 
	choices=["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"])
args = parser.parse_args()

# stories_file_name = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
data_type = args.type
data_corpus = args.corpus
story_path = "./data-tr/"
if data_type in ["10M", "100M"]:
	story_path += f"train_{data_type}/"
else:
	story_path += f"{data_type}/"
story_path += f"{data_corpus}.train" if data_type in ["10M", "100M"] else f"{data_corpus}.{data_type}"

window = 10
least_freq = 3
seed = 123
keep_stop_word = True
stop_word_as_anchor = False
preview = True
json_name = f"babylm-{data_type}-{data_corpus}.json"

# Import the sentence processor from spacy
nlp = spacy.load("en_core_web_sm", disable = ['parser'])
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')

def process_text(text, keep_stop_word=False, preview=True):
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
	return text


def process_file(file, keep_stop_word=True):
	print(f"Processing file: {file}")
	with open(file, 'r', encoding='latin-1') as fin:
		text = ""
		if data_corpus == "childes":
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
		output = process_text(text, keep_stop_word=keep_stop_word)
		return output


if not os.path.isfile(json_name):
	print("Reading and preprocessing...")
	story_tokens = process_file(story_path, keep_stop_word=keep_stop_word)
	with open(json_name, "w") as fout:
		json.dump(story_tokens, fout, indent=4)
else:
	print("Loading the story tokens...")
	with open(json_name, "r") as fin:
		story_tokens = json.load(fin)

# filter out the tokens with least frequencies:
overall_freq = Counter([word for word in story_tokens])
print("\nPreview of the 200 first words of the first preprocessed story:")
print(story_tokens[:200])
print("\nPreview of the 100 most frequent words:")
print(overall_freq.most_common(100))
print("\nPreview of the 100 least frequent words:")
print(overall_freq.most_common()[:-101:-1])
story_tokens = [word for word in story_tokens if overall_freq[word] >= least_freq]


