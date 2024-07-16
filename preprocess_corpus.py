import os
import re
from corpus_markers.gutenberg import IGNORE_LINE_MARKERS, NOVEL_START_MARKERS, STORY_START_MARKERS, NOVEL_END_MARKERS

def text_childes(file):
	text = ""
	for line in file:
		if "= = =" not in line.lower():
			prefix_pattern = re.compile(r'^\*[A-Z]+:\s*')
			line = prefix_pattern.sub("", line)
			text += line
	return text


def text_gutenberg(file, return_n_novels=False):
	"""
	References the cleaning implementation via https://github.com/c-w/gutenberg/
	markers: https://github.com/c-w/gutenberg/blob/master/gutenberg/_domain_model/text.py
	cleaning implementation: https://github.com/c-w/gutenberg/blob/master/gutenberg/cleanup/strip_headers.py
	"""
	text = ""
	header = ""
	header_lines = 0
	in_header = False
	in_footer = False
	weird_starrisk = False
	n_novels = 1
	numeral_pattern = re.compile(r'\d+$')

	for line in file:

		# If the line indicates the end of a novel
		if any(line.startswith(token) for token in NOVEL_END_MARKERS):
			# Ignore all the successive lines until a new novel begins.
			in_footer = True
			continue
		# elif the line indicates the start of a novel
		elif any(line.startswith(token) for token in NOVEL_START_MARKERS):
			# print(line)
			n_novels += 1
			in_header = True
			in_footer = False
			continue

		# elif any(line.startswith(token) for token in STORY_START_MARKERS):
		# 	start_novel_not_story = True
		# 	continue

		if in_header:
			# When the line is after the novel start but before the story start:
			header += line  # update the header
			header_lines += 1
			if numeral_pattern.search(line):
				# some line within the content
				continue
			if any(line.startswith(token) for token in STORY_START_MARKERS) or line.strip() in ("I", "I."):
				# if the current line indicates a potental story start:
				next_line = next(file).strip()
				# if next_line in header and header_lines < 50:
				if next_line in header:
					# if the text in the next line appears in the header, 
					# it is less likely that this is the real story start
					continue
				else:
					# find the real story start:
					header = ""  # reinitiate the header
					in_header = False
					continue

		# A further step detecting if the line within the story should be ignored:
		if any(line.startswith(token) for token in IGNORE_LINE_MARKERS):
			if "*       *       *" in line:
				weird_starrisk = True
			continue

		if weird_starrisk:
			weird_starrisk = False
			continue

		# The line within the story that should not be ignored:
		if not in_footer:
			text += line

	if return_n_novels:
		return text, n_novels
	else:
		return text


def text_open_subtitles(file):
	# In open_subtitles, some dialogues are all in uppercase,
	# and the tokenizer cannot lemmatize uppercases properly.
	text = ""
	for line in file:
		line = line.lower()
		pattern = re.compile(r'([,.!?;:])(?=\S)')  # find punctuation not followed by a space
		# then replace matched patterns with the punctuation followed by a space
		line = pattern.sub(r'\1', line)
		text += line
	return text


def text_simple_wiki(file):
	text = ""
	# Ignore the header of the file:
	in_header = True
	for line in file:
		if line.startswith("= = ="):
			in_header = False
			continue
		if not in_header:
			if len(line.strip().split()) == 1:
				# The section titles of the wiki term. Ignore them
				continue
			if line.startswith("= = ="):
				continue
			text += line
	return text


def text_switchboard(file):
	text = ""
	for line in file:
		text += line.lstrip("AB:")
	return text

