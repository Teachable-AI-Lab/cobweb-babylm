import os
import re
import copy
import sys
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from functools import reduce
from itertools import accumulate
import math
from typing import List, Optional, Union, Tuple
import numpy as np
from random import random

import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import bisect

from cobweb.cobweb import CobwebTree

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import MultiTokenEOSCriteria, stop_sequences_criteria

# Initialize tokenzier:
nlp = spacy.load("en_core_web_sm", disable = ['parser'])
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')

sys.stdout.reconfigure(encoding="utf-8")
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

import warnings

# Suppress specific warnings from the transformers library
def warning_filter(message, category, filename, lineno, file=None, line=None):
	if re.search(r"Token indices sequence length is longer than the specified maximum sequence length for this model \(\d+ > \d+\). Running this sequence through the model will result in indexing errors", str(message)):
		return False  # Suppress this warning
	return True  # Show other warnings
warnings.showwarning = warning_filter


@register_model("cobweb", "Cobweb", "Cobweb4L", "C4L", "c4l")
class CobwebLM(LM):
	"""
	Cobweb module that can be adopted to mainstream LM tasks.
	Can only make use of CPU in training. But may be capable in using multi-cores when testing.
	"""

	def __init__(
		self,
		window: int,
		load_file: str,
		# max_length: str = None,
		load_model: bool = True,
		bidirection: bool = False,
		keep_stop_word: bool = True,
		stop_word_as_anchor: bool = False,
		nodes_pred: int = 2000,
		used_cores: int = 120,
		device: str = "cpu",  # A required field in eval_lm arguments - we can just ignore it now
		tokenizer: str = "gpt2",
		batch_size: int = 64,  # A required field in eval_lm arguments - we can just ignore it now
		max_batch_size: int = 64,  # A required field in eval_lm arguments - we can just ignore it now
		decay: str = "inverse",  # the scheme of decay function applied to context words when generating an instance
		alpha_exp: float = 1.0,  # alpha value when doing decay function 'exp'
		sigma_gaussian: float = 1.0,  # sigma value when doing decay function 'gaussian'
		normalize: str = "softmax",  # the scheme of normalization applied to predicted probabilities of possible anchors when making predictions
		temperature: float = 1.0,  # the temperature when doing normalization. The distribution is sharper when temperature is higher
		# beam_search: bool = False,  # whether to do beam search when making a prediction
		# n_beam: int = 1,  # the number of beams when making a prediction through beam search
		token_form: str = "str",  # "str" or "encoding"
		prediction: str = "multi",  # "multi", "leaf", or "basic"
	) -> None:
		super().__init__()
		self._model = CobwebTree(0.000001, False, 0, True, False)
		self.window = window
		self.bidirection = bidirection
		if load_model:
			with open(load_file, 'r') as file:
				model_data = file.read()
			self._model.load_json(model_data)
		self.keep_stop_word = keep_stop_word
		self.stop_word_as_anchor = stop_word_as_anchor
		self.tokenizer = tokenizer
		self.nodes_pred = nodes_pred
		self.used_cores = used_cores
		# self.device = device
		self.temperature = temperature
		self.normalize_scheme = normalize
		self.decay_scheme = decay
		self.token_form = token_form
		# self.n_beam = n_beam


	def tokenize(self, text):
		if self.tokenizer == "spacy":
			punc = re.compile(r"[^_a-zA-Z0-9,.!?:;'\s]")
			whitespace = re.compile(r"\s+")
			text = re.sub(r'--', r' ', text)
			text = punc.sub("", text)
			text = whitespace.sub(" ", text)
			text = text.strip().lower()
			# parse:
			# print("\nThe text before preprocess (the first 200 words):")
			# print(' '.join(text.split()[:200]))
			text = nlp(text)
			if self.keep_stop_word:
				tokens = [token.lemma_.lower() for token in text if (not token.is_punct)]
			else:
				tokens = [token.lemma_.lower() for token in text if (not token.is_punct and not token.is_stop)]
		else:
			# tokenized with gpt-2 tokenizer
			if self.token_form == "str":
				# if so, generate the string tokens directly
				tokens = tokenizer_gpt2.tokenize(text)
			else:
				# generate the token ids instead
				tokens = [str(index) for index in tokenizer_gpt2.encode(text)]
		return tokens


	def text2instances(self, text):
		return list(self.tokens2instances(self, self.tokenize(text)))


	def tokens2instances(self, tokens):
		for anchor_idx, anchor_wd in enumerate(tokens):
			if self.stop_word_as_anchor:
				yield anchor_idx, self._tokens2instance(tokens, anchor_idx, anchor_wd)
			else:
				if anchor_wd not in nlp.Defaults.stop_words:
					yield anchor_idx, self._tokens2instance(tokens, anchor_idx, anchor_wd)


	def _tokens2instance(self, tokens, anchor_idx, anchor_wd):
		instance = {}
		window = self.window
		if self.bidirection:
			before_text = tokens[max(0, anchor_idx - window):anchor_idx]
			after_text = tokens[anchor_idx + 1:min(anchor_idx + 1 + window, len(tokens))]
			context_before = {}
			context_after = {}
			# In a language task, the context words are not considered as simple counts.
			# Considering the proximity to the anchor word, the further the context word to the anchor, the less weight it will have
			for i, w in enumerate(before_text):
				distance = len(before_text) - i
				context_before[w] = self._compute_weight(distance)
				# context_before[w] = 1 / abs(len(before_text) - i)
			for i, w in enumerate(after_text):
				distance = i + 1
				context_after[w] = self._compute_weight(distance)
				# context_after[w] = 1 / (i + 1)

			instance['context-before'] = context_before
			instance['context-after'] = context_after

		else:
			before_text = tokens[max(0, anchor_idx - window):anchor_idx]
			context = {}
			for i, w in enumerate(before_text):
				distance = len(before_text) - i
				context[w] = self._compute_weight(distance)
				# context[w] = 1 / abs(len(before_text) - i)
			instance['context'] = context

		if anchor_wd is None:
			return instance
		instance['anchor'] = {anchor_wd: 1}
		return instance


	def _compute_weight(self, distance):
		"""
		The function to calculate the corresponding weight for a context word
		given its proximity to the anchor word and the chosen function.
		"""
		def weight_inverse(d):
			return 1 / abs(d)

		def weight_linear(d):
			return 1 / (abs(d) + 1)

		def weight_exp(d):
			return np.exp(-self.alpha_exp * abs(d))

		def weight_gaussian(d):
			return np.exp(-(d ** 2) / (2 * self.sigma_gaussian ** 2))

		weight_funcs = {
			"inverse": weight_inverse,
			"linear": weight_linear,
			"exp": weight_exp,
			"gaussian": weight_gaussian
			}
		weight_func = weight_funcs.get(self.decay_scheme, weight_inverse)
		return weight_func(distance)


	def context2instance(self, context_tokens):
		instance = {}
		# context_tokens = self.tokenize(context_str)
		context_dict = {}

		for i, w in enumerate(context_tokens):
			# if '"' in w:
			# 	w = w.replace('"', "''")
			context_dict[w] = 1 / abs(len(context_tokens) - i)
		instance['context'] = context_dict
		return instance


	def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
		"""
		Instance.args: Tuple[str, str]:
			str: an input string to the LM
			str: a target string on which the logLL of the LM producing this target, 
				conditioned on the input, will be returned
		Each request returns `(ll, is_greedy)`:
			ll: a floating point number representing the log prob of generating the target string conditioned on the input
			is_greedy: 0 or 1. It is 1 iff the target string would be generated by greedy sampling from the LM
				(if the target string is the most likely N-token string to be output by the LM given the input)
		"""
		# new_reqs = []  # preprocessed requests - make the strings "encoded" - i.e. make them into instances
		# contexts_insts = []
		# continuations_tokens = []
		# for context, continuation in [req.args for req in requests]:
		# 	# Transform each context string to corresponding instance
		# 	contexts_insts.append(self.context2instance(self.tokenize(context)))
		# 	# Transform each continuation string into tokens
		# 	continuations_tokens.append(self.tokenize(continuation))

		# # Generate probs_pred for all context instances:
		# # context_probs_pred = self.model.predict_probs_parallel(contexts_insts, self.nodes_pred, False, False, self.used_cores)

		# for i in range(len(contexts_probs_pred)):
		# 	yield self._loglikelihood_iter(contexts_insts[i], continuations_tokens[i])


		# context_insts = [self.context2instance(context) for context in ]
		# for context, continuation in [req.args for req in requests]:
		# 	if context == "":
		# 		# End of text as context:
		# 		context_enc, continuation_enc = ["EOT"], self.text2instances(continuation)
		# 	else:
		# 		context_enc, continuation_enc = self.text2instances(context), self.text2instances(continuation)
		# 	new_reqs.append(((context, continuation), context_enc, continuation_enc))
		

		# for req in requests:
		# 	yield self._loglikelihood_iter(req)

		# print(type(requests[0]))
		for context, continuation in [req.args for req in requests]:
			yield self._loglikelihood_iter(self.tokenize(context), self.tokenize(continuation)), True


	def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
		"""
		Instance.args: Tuple[str]:
			str: an input string whose entire logLL conditioned on purely the EOT (End-of-Text) token will be calculated
		Used to evaluate perplexity on a data distribution.
		Each request returns (ll, ): Tuple[float]. Solely the logLL of producing each piece of text given no starting input.

		This is a special case of CobwebLM.loglikelihood.
		"""
		context = []
		for req in requests:
			yield self._loglikelihood_iter(context, self.tokenize(req.args[0]))
		pass


	def _loglikelihood_iter(self, context_tokens, continuation_tokens):

		if len(context_tokens) > 0:
			context_insts = [self.context2instance(context_tokens)]
			context_tokens_iter = context_tokens.copy()
		else:
			# NO context - EOT (End of text)
			context_insts = []
			context_tokens_iter = []

		# First generate the batch of context instances generated from continuation tokens (additional one for each)
		for token in continuation_tokens:
			context_tokens_iter.append(token)
			context_insts.append(self.context2instance(context_tokens_iter))

		# Generate the probs_pred for all context instances:
		probs_pred = self._model.predict_probs_parallel(context_insts, self.nodes_pred, False, False, self.used_cores)
		probs_pred = [prob_pred["anchor"] for prob_pred in probs_pred]

		# Then normalize the predictions and find the one with the greatest normalized probability:
		probs_pred_normalized = [self._normalized_prob_pred(prob_pred) for prob_pred in probs_pred]

		# Then compute the probabilities for each new token given the updated context instance:
		# So basically here are the p(xn|x1, ..., xn-1)'s
		probs_pred_tokens = [prob_pred[token] if token in prob_pred else 0. for token in continuation_tokens for prob_pred in probs_pred_normalized]
		probs_pred_tokens_log = [math.log(prob) if prob != 0. else prob for prob in probs_pred_tokens]

		# Finally generate the overall probability of the continuation:
		# It is basically an autoregressive way of calculating:
		# P(w1|ctxt) P(w2|ctxt, w1) P(w3|ctxt, w1, w2) ...
		# return reduce(lambda x, y: x * y, probs_pred_tokens)
		# return list(accumulate(probs_pred_tokens_log))
		return sum(probs_pred_tokens_log)


	def _normalized_prob_pred(self, prob_pred):

		anchor_options = list(prob_pred.keys())
		anchor_prob = np.array(list(prob_pred.values()))

		# print("\nTest normalize function:")
		# print(anchor_options[:5])
		# print(anchor_prob[:5])
		# for option in anchor_options[:5]:
		# 	print(option, prob_pred[option])

		anchor_prob_normalized = self._normalize(logits=anchor_prob)
		prob_pred_normalized = {}
		for i in range(len(anchor_options)):
			prob_pred_normalized[anchor_options[i]] = anchor_prob_normalized[i]
		return prob_pred_normalized


	def _normalize(self, logits):
		logits = logits / self.temperature

		def softmax(logits):
			exp_logits = np.exp(logits - np.max(logits))
			return exp_logits / np.sum(exp_logits)

		def sigmoid(logits):
			return 1 / (1 + np.exp(-logits))

		def min_max(logits):
			return (logits - np.min(logits)) / (np.max(logits) - np.min(logits))

		def z_score(logits):
			return (logits - np.mean(logits)) / np.std(logits)

		def l2(logits):
			norm = np.linalg.norm(logits)
			return logits / norm if norm != 0 else logits

		normalize_funcs = {
			"softmax": softmax, "sigmoid": sigmoid, "min_max": min_max, 
			"z_score": z_score, "l2": l2
			}
		normalize_func = normalize_funcs.get(self.normalize_scheme, softmax)
		return normalize_func(logits)


	def generate_until(self, requests: List[Instance]) -> List[str]:
		"""
		Instance.args: Tuple[str, dict]:
			str: input string
			dict: a dictionary of keyword arguments used to control generation parameters.
		Text will be sampled from the LM with the arguments.
		For example, until a maximum output length or specific stopping string sequences:
			dict = {"until": ["\n\n", "."], 
					"max_gen_toks": 128, 
					"method": "multi" or "leaf" or "basic", 
					"n_beam": 1}
		Each request returns the input+output text from the model.
		"""
		for context, args in [req.args for req in requests]:
			# method = args["method"]
			n_beam = args["n_beam"] if "n_beam" in args else 1
			if n_beam == 1:
				yield self._generate_iter(context, args)
			else:
				yield self._generate_iter_beams(context, args, n_beam=n_beam)


	def _generate_iter(self, context_str, args_dict):
		"""
		One problem - there are only lemmatized tokens in the Cobweb tree,
		so only the lemmatized tokens will be generated.
		So maybe we need to include unlemmatized tokens as well as punctuations.
		"""
		end_tokens = args_dict['until'] if "until" in args_dict else None
		max_length = args_dict['max_gen_toks'] if "max_gen_toks" in args_dict else None
		method = args_dict['method'] if "method" in args_dict else None  # "multi", "leaf", or "basic"
		new_tokens = []
		new_token = ""
		valid_condition = new_token not in end_tokens and len(new_tokens) < max_length
		context_tokens = self.tokenize(context_str)
		# print(valid_condition)

		while valid_condition:
			context_inst = self.context2instance(context_tokens)
			new_token = self._predict_next_token(context_inst, method=method)
			new_tokens.append(new_token)
			context_tokens.append(new_token)
			valid_condition = new_token not in end_tokens and len(new_tokens) < max_length
			# print(valid_condition)

		if self.token_form == "str" or self.tokenizer == "spacy":
			new_str = " ".join(new_tokens)
			return context_str + " " + new_str
		else:
			# tokenized by gpt-2 and use encodings
			complete_tokens = context_tokens + new_tokens
			return tokenizer_gpt2.decode([int(index) for index in complete_tokens])
		# 	new_str = tokenizer_gpt2.decode(new_tokens)
		# return context_str + " " + new_str


	def _generate_iter_beams(self, context_str, args_dict, n_beam):
		end_tokens = args_dict['until'] if "until" in args_dict else None
		max_length = args_dict['max_gen_toks'] if "max_gen_toks" in args_dict else None
		method = args_dict['method'] if "method" in args_dict else None  # "multi", "leaf", or "basic"
		# new_tokens = []
		# new_token = ""
		# valid_condition = new_token not in end_tokens and len(new_tokens) < max_length
		context_tokens_ori = self.tokenize(context_str)
		n_context_tokens_ori = len(context_tokens_ori)

		previous_best_tokens = []  # List[(list of tokens, perplexity)], the last one has the greatest perplexity
		current_best_tokens = []  # List[(list of tokens, perplexity)], the last one has the greatest perplexity

		def _valid_condition(tokens):
			# Keep verify if the current best generated token is in the end_tokens,
			# or if the length of generated tokens have reached the limit
			return tokens[-1] not in end_tokens and len(tokens) < n_context_tokens_ori + max_length

		def _maintain_sorted_list(new_tuple, sorted_list, max_length):
			"""
			Used as the sorted list for storing the best current perplexities so far.
			"""
			if len(sorted_list) < max_length or new_tuple[1] > sorted_list[0][1]:
				if len(sorted_list) >= max_length:
					sorted_list.pop(0)
				pos = bisect.bisect_left([x[1] for x in sorted_list], new_tuple[1])
				sorted_list.insert(pos, new_tuple)
				# bisect.insort(sorted_list, new_tuple, key=lambda x: x[1])

		while len(previous_best_tokens) == 0 or _valid_condition(current_best_tokens[-1][0]):
			# New iteration for the next new token.

			if len(previous_best_tokens) == 0:
				# The first iteration. Just include the initial context tokens:
				previous_best_tokens.append((context_tokens_ori, 0.))
				# Here current_best_tokens should be an empty list
			else:
				# Not the first iteration.
				previous_best_tokens = current_best_tokens.copy()
				current_best_tokens = []

			for context_tokens, perplexity_ctx in previous_best_tokens:
				# Find the n_beam best new token for each possible previous generated tokens:
				context_inst = self.context2instance(context_tokens)
				possible_new_tokens = self._predict_next_token_beam(context_inst, n_beam, method)
				for possible_new_token, logll in possible_new_tokens:
					# First compute the perplexity of the context tokens with the possible new token:
					perplexity_current = perplexity_ctx + logll
					# Update the sorted current best tokens list:
					_maintain_sorted_list(
						(context_tokens + [possible_new_token], perplexity_current),
						current_best_tokens,
						max_length=n_beam,
					)

		# Now the current_best_tokens with n_beam options is returned.
		new_tokens = current_best_tokens[-1][0]
		if self.token_form == "str" or self.tokenizer == "spacy":
			new_str = " ".join(new_tokens)
			return context_str + " " + new_str
		else:
			# tokenized by gpt-2 and use encodings
			complete_tokens = context_tokens_ori + new_tokens
			return tokenizer_gpt2.decode([int(index) for index in complete_tokens])


	def _predict_next_token(self, inst, method="multi"):
		"""
		Given an instance with no anchor word, predict the next anchor word
		"""
		if method == "multi":
			prob_pred = self._model.predict_probs(inst, self.nodes_pred, False, False)["anchor"]
			# prob_pred = self._normalized_prob_pred(prob_pred)
			# next_token = sorted([(prob_pred[option], random(), option) for option in prob_pred.keys()], reverse=True)[0][2]
		else:
			# Single-node prediction. Either leaf or basic-level node
			node = self._model.categorize(inst)
			if method == "basic":
				node = node.get_basic_level()
			prob_pred = node.predict_probs()["anchor"]
		prob_pred = self._normalized_prob_pred(prob_pred)
		next_token = sorted([(prob_pred[option], random(), option) for option in prob_pred.keys()], reverse=True)[0][2]
		return next_token


	def _predict_next_token_beam(self, inst, n_beam, method):
		"""
		Given an instance with no anchor word, return the next n best anchor word 
		(and maybe n can be the number of beams though they are not necessary to be any of the best n beams)
		"""
		if method == "multi":
			prob_pred = self._model.predict_probs(inst, self.nodes_pred, False, False)["anchor"]
		else:
			# Single-node prediction. Either leaf or basic-level node
			node = self._model.categorize(inst)
			if method == "basic":
				node = node.get_basic_level()
			prob_pred = node.predict_probs()["anchor"]
		prob_pred = self._normalized_prob_pred(prob_pred)
		# Then find the best n next tokens:
		possible_new_tokens = sorted([(prob_pred[option], random(), option) for option in prob_pred.keys()], reverse=True)[:n_beam]
		return [(new_token_tuple[2], new_token_tuple[0]) for new_token_tuple in possible_new_tokens]


