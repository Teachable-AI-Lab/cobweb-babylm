# Cobweb in Language Modeling

This repo includes:
- How to train Cobweb suitable for language modeling in general
- How trained Cobweb can be evaluated with the LM evaluation pipeline with bundled benchmarks (employed by BabyLM Challenge as well)
Related link: [BabyLM Challenge](https://babylm.github.io/index.html)

### To Train a Cobweb Tree


#### Download Training Data

You first need to download the datasets and put them under `./data-tr` in their corresponding location. The folders under `./data-tr` are just placeholders. Here the following datasets should be included in the repo:
- BabyLM Challenge data (`https://osf.io/ad7qg/`). As for 2024, the BabyLM Challenge provides text training data, multimodal training data, and partial evaluation data. Here we include the text training data only, and please download the `text_data` folder with the sharing link provided. The text data includes the following types of data:
	- 10M training datasets (`train_10M`). Five corpuses (`bnc_spoken`, `childes`, `gutenberg`, `open_substitles`, `simple_wiki`, and `switchboard`) are provided and they comparise to about a vocab size of 10 million.
	- 100M training datasets (`train_100M`). The same five corpuses as forementioned but with more contents, and they comparise to about 100 million words in total in the vocabulary.
	- Validation dataset (`dev`). The same five corpuses but for validation.
	- Test dataset (`test`). The same five corpuses but for test.
- Sherlock Holmes stories text data. Being used in the [MSR Sentence Completion Challenge](https://www.microsoft.com/en-us/research/project/msr-sentence-completion-challenge/overview/). Can be download [here](https://github.com/Teachable-AI-Lab/cobweb-language/tree/main/data-tr).


#### Train the Tree

Train the Cobweb tree with

	python3 cobweb-tr.py [--arguments]

with a couple of argument options. Here I list some of the most curcial arguments:

- `--type`: The type of training data used. Available options: `10M`, `100M`, `dev`, `test`, `holmes`.
- `--corpus`: The corpus used in the training data (if `--type=holmes` you can ignore this argument). Available options: `all` (all six corpuses under the training data type), `bnc_spoken`, `childes`, `gutenberg`, `open_subtitles`, `simple_wiki`, `switchboard`.
- `--tokenizer`: The tokenizer used. Can be either `gpt2` or `spacy`.
- `--scheme`: The distance/dissimilarity function used in calculation. Available options: `inverse`, `linear`, `exp`, `gaussian`.
- `--window`: The context window size (`int`) in every transformed trained instance.
- `--n-tr-splits`: The number of training splits/checkpoints in the training process (`int`)
- `--load-model`: If included, the model will seek for the model file stored in the argument `--load-model-file` before training.
- `--token-form`: The data type for the attributes stored in each transformed trained instance. Available options: `str` and `encoding`. If `--tokenizer=gpt2`, I STRONGLY recommeded using `encoding`.

For more details just see the original Python script `cobweb-tr.py` or enter `python3 cobweb-tr.py --h`.


#### Evaluation

To evaluate the model, we use the evaluation pipeline `lm-eval` used by the BabyLM Challenge. The detail of the pipeline is available [here](https://github.com/EleutherAI/lm-evaluation-harness). The BabyLM evaluation repo (folked from the original `lm-eval`) is available [here](https://github.com/babylm/evaluation-pipeline-2024). The evaluation pipeline used here is at `./lm-evaluation-harness`, and please check `./lm-evaluation-harness/README.md` for more details.







