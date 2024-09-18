import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time

# Utils
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

class PunktLanguageVars:
    """
    Stores variables, mostly regular expressions, which may be
    language-dependent for correct application of the algorithm.
    An extension of this class may modify its properties to suit
    a language other than English; an instance can then be passed
    as an argument to PunktSentenceTokenizer and PunktTrainer
    constructors.
    """

    __slots__ = ("_re_period_context", "_re_word_tokenizer")

    def __getstate__(self):
        # All modifications to the class are performed by inheritance.
        # Non-default parameters to be pickled must be defined in the inherited
        # class.
        return 1

    def __setstate__(self, state):
        return 1

    sent_end_chars = (".", "?", "!")
    """Characters which are candidates for sentence boundaries"""

    @property
    def _re_sent_end_chars(self):
        return "[%s]" % re.escape("".join(self.sent_end_chars))

    internal_punctuation = ",:;"  # might want to extend this..
    """sentence internal punctuation, which indicates an abbreviation if
    preceded by a period-final token."""

    re_boundary_realignment = re.compile(r'["\')\]}]+?(?:\s+|(?=--)|$)', re.MULTILINE)
    """Used to realign punctuation that should be included in a sentence
    although it follows the period (or ?, !)."""

    _re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
    """Excludes some characters from starting word tokens"""

    @property
    def _re_non_word_chars(self):
        return r"(?:[)\";}\]\*:@\'\({\[%s])" % re.escape(
            "".join(set(self.sent_end_chars) - {"."})
        )

    """Characters that cannot appear within words"""

    _re_multi_char_punct = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)"
    """Hyphen and ellipsis are multi-character punctuation"""

    _word_tokenize_fmt = r"""(
        %(MultiChar)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
        (?= # Sequences marking a word's end
            \s|                                 # White-space
            $|                                  # End-of-string
            %(NonWord)s|%(MultiChar)s|          # Punctuation
            ,(?=$|\s|%(NonWord)s|%(MultiChar)s) # Comma if at end of word
        )
        |
        \S
    )"""

    """Format of a regular expression to split punctuation from words,
    excluding period."""

    def _word_tokenizer_re(self):
        """Compiles and returns a regular expression for word tokenization"""
        try:
            return self._re_word_tokenizer
        except AttributeError:
            self._re_word_tokenizer = re.compile(
                self._word_tokenize_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "MultiChar": self._re_multi_char_punct,
                    "WordStart": self._re_word_start,
                },
                re.UNICODE | re.VERBOSE,
            )
            return self._re_word_tokenizer

    def word_tokenize(self, s):
        """Tokenize a string to split off punctuation other than periods"""
        return self._word_tokenizer_re().findall(s)

    _period_context_fmt = r"""
        %(SentEndChars)s(?!(\" ))
                  # a potential sentence ending
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            \s+(?P<next_tok>\S+)     # or whitespace and some other token
        )
                       # negative lookahead for quotation mark followed by space
      )"""
    """Format of a regular expression to find contexts including possible
    sentence boundaries. Matches token which the possible sentence boundary
    ends, and matches the following token within a lookahead expression."""

    def period_context_re(self):
        """Compiles and returns a regular expression to find contexts
        including possible sentence boundaries."""
        try:
            return self._re_period_context
        except:
            self._re_period_context = re.compile(
                self._period_context_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "SentEndChars": self._re_sent_end_chars,
                },
                re.UNICODE | re.VERBOSE,
            )
           
            return self._re_period_context

_re_non_punct = re.compile(r"[^\W\d]", re.UNICODE)

def save_data_to_json(data, directory_path, file_name, is_metadata = False):
    """
    Save the list of dictionaries to a JSON file in the specified directory.

    Args:
        data (list): List of dictionaries to be saved.
        directory_path (str): Path to the directory where the file will be saved.
        file_name (str): Name of the JSON file.

    Returns:
        str: The path to the saved JSON file.
    """
    def convert_to_python_int(data):
        if isinstance(data, list):
            return [convert_to_python_int(item) for item in data]
        elif isinstance(data, dict):
            return {key: convert_to_python_int(value) for key, value in data.items()}
        elif isinstance(data, (np.integer)):
            return int(data)
        elif isinstance(data, (np.floating)):
            return float(data)
        else:
            return data
        
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            return super(NumpyEncoder, self).default(obj)

    def make_serializable(data):
        if isinstance(data, dict):
            return {k: make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [make_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.floating, np.ndarray, np.generic)):
            return json.loads(json.dumps(data, cls=NumpyEncoder))
        return data

    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, file_name)

    if is_metadata:
        # Convert data to be JSON serializable
        try:
            serializable_data = make_serializable(data)
        except (TypeError, OverflowError) as e:
            logger.error(f"Error serializing data: {e}")
            return False
        
        # Write data to file with error handling
        try:
            with open(file_path, 'w') as json_file:
                json.dump(serializable_data, json_file, indent=4, cls = NumpyEncoder)
        except (TypeError, IOError) as e:
            logger.error(f"Error writing to file: {e}")
            return False
        
        # Verify file integrity by reading it back
        try:
            with open(file_path, 'r') as json_file:
                loaded_data = json.load(json_file)
                print(loaded_data == serializable_data) 
        except (IOError, json.JSONDecodeError, AssertionError) as e:
            logger.error(f"Error verifying file integrity: {e}")
            return False
        logger.info(f'Data saved to {file_path}')
        return True
    else:
        try:
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        except IOError as e:
            logger.error(f"Error writing to file: {e}")
            return False
        logger.info(f'Data saved to {file_path}')
        return True