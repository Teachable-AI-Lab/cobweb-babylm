import hashlib
import json
# import openai
import os
import pickle
import pytest
import unittest.mock as mock
from typing import List, Optional, Union, Tuple

import lm_eval.models as models
from lm_eval.api.instance import Instance
from lm_eval.models.cobweb_lm import CobwebLM


def tuple_to_instance(t: Tuple) -> Instance:
    return Instance(
        request_type=t[0],
        doc=t[1],
        arguments=t[2],
        idx=t[3],
        metadata=t[4],
        resps=t[5],
        filtered_resps=t[6]
    )

def tuples_to_instance(ts, function, idx, task, doc):
    instances = []
    for t in ts:
        instances.append(
            tuple_to_instance((function, {}, t, idx, (task, doc, idx), [], {}))
        )
    return instances



LOGLIKELIHOOD_TEST_CASES = [
    ("The quick brown fox jumps over the lazy", " dog"),
    ("The quick brown fox jumps over the lazy", " cat"),
    ("The quick brown fox jumps over the lazy", ", lazy dog"),
    ("The quick brown fox jumps over the lazy", ", lazy fox"),
    (
        "The quick brown fox jumps over the lazy",
        ", lazy fox and they both fall to the ground",
    ),
    (
        """A mult""",
        """ilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN)""",
    ),
    (
        """The term MLP is used ambiguously, sometimes loosely to any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons""",
        """ (with threshold activation); see § Terminology""",
    ),
    (
        """Multilayer perceptrons are sometimes coll""",
        """oquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.[1]""",
    ),
    (
        """An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear""",
        """ activation function.""",
    ),
    (
        """MLP utilizes a supervised""",
        """ learning technique called backpropagation for training.[2][3] Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.[4]""",
    ),
    (
        """Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic""",
        """ in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. """,
    ),
    (
        """Specifically, we train GPT-3, an autoregressive language model with 175""",
        """ billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.""",
    ),
    (
        """A mult""",
        """ilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN)""",
    ),
    ("""Hello""", """ World"""),
]


load_model = True
window = 10
bidirection = False
keep_stop_word = True
stop_word_as_anchor = True
# load_file = "~/cobweb-babylm/model-saved/cobweb/cobweb-data10M-bnc_spoken-seed123-window10-leastfreq1-instances337768-split0-05212024.json"
# load_file = "../../model-saved/cobweb/cobweb-data10M-bnc_spoken-seed123-window10-leastfreq1-instances971023-split0-05262024.json"
# load_file = "../model-saved/cobweb/cobweb-data10M-switchboard-tokenizer_gpt2-seed123-window10-leastfreq1-instances1000-split0-06062024.json"
# load_file = "../model-saved/cobweb/cobweb-data10M-gutenberg-tokenizer_gpt2-seed123-window10-leastfreq1-instances3551073-split0-06052024.json"
# load_file = "../model-saved/cobweb/cobweb-data10M-all-tokenizer_gpt2-scheme_inverse-seed123-window10-leastfreq1-instances1189512-split4-06232024-vocab45034.json"
load_file = "../model-saved/cobweb/cobweb-data10M-all-tokenizer_gpt2-token_form_encoding-scheme_inverse-seed123-window10-leastfreq1-instances1189512-split8-06262024-vocab45044.json"
max_length = 20
nodes_pred = 1000
used_cores = 120
device = "cpu"

# print("fine1")

model = CobwebLM(load_model=load_model, window=window, bidirection=bidirection, 
    keep_stop_word=keep_stop_word, stop_word_as_anchor=stop_word_as_anchor, 
    load_file=load_file, nodes_pred=nodes_pred, 
    used_cores=used_cores, device=device, token_form="encoding")

# print("fine2")

"""
Three functionalities to test:
loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]
loglikelihood_rolling(self, requests: List[Instance]) -> List[float]
generate_until(self, requests: List[Instance]) -> List[str]

"""


# loglikelihood:
print("======== TEST LOGLIKELIHOOD ========")
instances = tuples_to_instance(LOGLIKELIHOOD_TEST_CASES, function="loglikelihood", idx=0, task="task1", doc="doc1")
ll_results = model.loglikelihood(instances)
ll_results_ls = list(ll_results)
# print(list(ll_results))
# ll_results = [(context, target, logll, greedy) for logll, greedy in ll_results]
print(len(LOGLIKELIHOOD_TEST_CASES), len(ll_results_ls))
for i in range(len(LOGLIKELIHOOD_TEST_CASES)):
    context, target = LOGLIKELIHOOD_TEST_CASES[i]
    print("\n")
    print(context)
    print(target)
    logll, greedy = ll_results_ls[i]
    print(logll, greedy)


# loglikelihood_rolling:
print("======== TEST LOGLIKELIHOOD ROLLING ========")
test_string = "We study empirical scaling laws for language model performance on the cross-entropy loss."
instances = tuples_to_instance([(test_string,)], function="loglikelihood_rolling", idx=0, task="task1", doc="doc1")
perplexities = model.loglikelihood_rolling(instances)
perplexities = [(context, logll) for logll in perplexities]
for i in range(len(perplexities)):
    context = test_string
    logll = list(perplexities)[i]
    print("\n")
    print(context)
    print(logll)


# generate_until:
print("======== TEST GENERATE UNTIL ========")
# requests = [("wonderful", {"until": "_stop", "max_gen_toks": max_length}), 
#             ("play", {"until": "_do", "max_gen_toks": max_length})]
requests = [("Do you know where is the Howard CTA station?", 
            {"max_gen_toks": max_length, "until": ["_stop"], "n_beam": 3, "method": "multi"}), 
            ("Today I'd like to introduced Jimmy Carter, who is the representative of GA Department.", 
            {"max_gen_toks": max_length, "until": ["_stop"], "n_beam": 3, "method": "multi"}),
            ("Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic",
            {"max_gen_toks": max_length, "until": ["_stop"], "n_beam": 3, "method": "multi"}),
            ("It's me, Mary. How can I help you with your car?",
            {"max_gen_toks": max_length, "until": ["_stop"], "n_beam": 3, "method": "multi"})
            ]
instances = tuples_to_instance(requests, function="generate_until", idx=0, task="task1", doc="doc1")
strings_generated = model.generate_until(instances)
for string_generated in strings_generated:
    print("\n")
    print(string_generated)
    print(len(list(string_generated.split())))







