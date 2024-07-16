"""Data module that contains strings that mark the start and end of a Project
Gutenberg disclaimer/header."""


"""
References the cleaning implementation via https://github.com/c-w/gutenberg/
markers: https://github.com/c-w/gutenberg/blob/master/gutenberg/_domain_model/text.py
cleaning implementation: https://github.com/c-w/gutenberg/blob/master/gutenberg/cleanup/strip_headers.py
"""

# TEXT_START_MARKERS = frozenset((
#     "*END*THE SMALL PRINT",
#     "*** START OF THE PROJECT GUTENBERG",
#     "*** START OF THIS PROJECT GUTENBERG",
#     "This etext was prepared by",
#     "E-text prepared by",
#     "Produced by",
#     "Distributed Proofreading Team",
#     "Proofreading Team at http://www.pgdp.net",
#     "http://gallica.bnf.fr)",
#     "      http://archive.org/details/",
#     "http://www.pgdp.net",
#     "by The Internet Archive)",
#     "by The Internet Archive/Canadian Libraries",
#     "by The Internet Archive/American Libraries",
#     "public domain material from the Internet Archive",
#     "Internet Archive)",
#     "Internet Archive/Canadian Libraries",
#     "Internet Archive/American Libraries",
#     "material from the Google Print project",
#     "*END THE SMALL PRINT",
#     "***START OF THE PROJECT GUTENBERG",
#     "This etext was produced by",
#     "*** START OF THE COPYRIGHTED",
#     "The Project Gutenberg",
#     "http://gutenberg.spiegel.de/ erreichbar.",
#     "Project Runeberg publishes",
#     "Beginning of this Project Gutenberg",
#     "Project Gutenberg Online Distributed",
#     "Gutenberg Online Distributed",
#     "the Project Gutenberg Online Distributed",
#     "Project Gutenberg TEI",
#     "This eBook was prepared by",
#     "http://gutenberg2000.de erreichbar.",
#     "This Etext was prepared by",
#     "This Project Gutenberg Etext was prepared by",
#     "Gutenberg Distributed Proofreaders",
#     "Project Gutenberg Distributed Proofreaders",
#     "the Project Gutenberg Online Distributed Proofreading Team",
#     "**The Project Gutenberg",
#     "*SMALL PRINT!",
#     "More information about this book is at the top of this file.",
#     "tells you about restrictions in how the file may be used.",
#     "l'authorization à les utilizer pour preparer ce texte.",
#     "of the etext through OCR.",
#     "*****These eBooks Were Prepared By Thousands of Volunteers!*****",
#     "We need your donations more than ever!",
#     " *** START OF THIS PROJECT GUTENBERG",
#     "****     SMALL PRINT!",
#     '["Small Print" V.',
#     '      (http://www.ibiblio.org/gutenberg/',
#     'and the Project Gutenberg Online Distributed Proofreading Team',
#     'Mary Meehan, and the Project Gutenberg Online Distributed Proofreading',
#     '                this Project Gutenberg edition.',
# ))


# TEXT_END_MARKERS = frozenset((
#     "*** END OF THE PROJECT GUTENBERG",
#     "*** END OF THIS PROJECT GUTENBERG",
#     "***END OF THE PROJECT GUTENBERG",
#     "End of the Project Gutenberg",
#     "End of The Project Gutenberg",
#     "Ende dieses Project Gutenberg",
#     "by Project Gutenberg",
#     "End of Project Gutenberg",
#     "End of this Project Gutenberg",
#     "Ende dieses Projekt Gutenberg",
#     "        ***END OF THE PROJECT GUTENBERG",
#     "*** END OF THE COPYRIGHTED",
#     "End of this is COPYRIGHTED",
#     "Ende dieses Etextes ",
#     "Ende dieses Project Gutenber",
#     "Ende diese Project Gutenberg",
#     "**This is a COPYRIGHTED Project Gutenberg Etext, Details Above**",
#     "Fin de Project Gutenberg",
#     "The Project Gutenberg Etext of ",
#     "Ce document fut presente en lecture",
#     "Ce document fut présenté en lecture",
#     "More information about this book is at the top of this file.",
#     "We need your donations more than ever!",
#     "END OF PROJECT GUTENBERG",
#     " End of the Project Gutenberg",
#     " *** END OF THIS PROJECT GUTENBERG",
# ))


# LEGALESE_START_MARKERS = frozenset(("<<THIS ELECTRONIC VERSION OF",))


# LEGALESE_END_MARKERS = frozenset(("SERVICE THAT CHARGES FOR DOWNLOAD",))


# The following does not refer to anyone's work

IGNORE_LINE_MARKERS = frozenset((
    "*CHAPTER",
    "[Illustration:",
    "= = =",
    "CHAPTER",
    "JANUARY", "FEBURARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    "*       *       *",

    ))


NOVEL_START_MARKERS = frozenset((
    "= = =",
    "= = = PG"))


STORY_START_MARKERS = frozenset((
    "Boston, Mass., U. S. A.",
    "CHAPTER I",
    "JANUARY 1:",
    "I.",
    "PART I",
    "CHAPTER ONE",
    "PROLOGUE",
    "_No. 1.",
    "[Illustration",
    "FIRST",
    "_Chapter I_",
    "ONLY A NURSE GIRL!",
    "1 ",

    ))


NOVEL_END_MARKERS = frozenset((
    "THE END.",
    "Transcriber Notes",
    ))



