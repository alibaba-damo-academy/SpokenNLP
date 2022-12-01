# coding=utf-8
# Copyright 2022 Alibaba.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unicodedata
import sys
import os
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
from tokenizer_utils import convert_to_unicode, whitespace_tokenize, _is_whitespace, _is_control, _is_punctuation, _is_symbol

class BasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (bool):
            Whether to lowercase the input when tokenizing.
            Defaults to `True`.
        never_split (Iterable):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (bool):
            Whether to tokenize Chinese characters.
        strip_accents: (bool):
            Whether to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """

    def __init__(self,
                 do_lower_case=True,
                 never_split=None,
                 tokenize_chinese_chars=True,
                 strip_accents=None):
        """Constructs a BasicTokenizer."""
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Tokenizes a piece of text using basic tokenizer.

        Args:
            text (str): A piece of text.
            never_split (List[str]): List of token not to split.

        Returns:
            list(str): A list of tokens.

        Examples:
            .. code-block::

                from paddlenlp.transformers import BasicTokenizer
                basictokenizer = BasicTokenizer()
                tokens = basictokenizer.tokenize('He was a puppeteer')
                '''
                ['he', 'was', 'a', 'puppeteer']
                '''
        """
        text = convert_to_unicode(text)
        never_split = self.never_split.union(
            set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens


    def _run_strip_accents(self, text):
        """
        Strips accents from a piece of text.
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """
        Splits punctuation on a piece of text.
        """
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # punctuation and symbol should be treat as single char.
            if _is_punctuation(char) or _is_symbol(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """
        Adds whitespace around any CJK character.
        """
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """
        Checks whether CP is the codepoint of a CJK character.
        """
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """
        Performs invalid character removal and whitespace cleanup on text.
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)



class WordpieceTokenizer(object):
    """
    Runs WordPiece tokenization.

    Args:
        vocab (Vocab|dict):
            Vocab of the word piece tokenizer.
        unk_token (str):
            A specific token to replace all unknown tokens.
        max_input_chars_per_word (int):
            If a word's length is more than
            max_input_chars_per_word, it will be dealt as unknown word.
            Defaults to 100.
    """

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer`.

        Returns:
            list (str): A list of wordpiece tokens.

        Examples:
            .. code-block::

                from paddlenlp.transformers import BertTokenizer, WordpieceTokenizer

                berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                vocab  = berttokenizer.vocab
                unk_token = berttokenizer.unk_token

                wordpiecetokenizer = WordpieceTokenizer(vocab,unk_token)
                inputs = wordpiecetokenizer.tokenize("unaffable")
                print(inputs)
                '''
                ["un", "##aff", "##able"]
                '''
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


if __name__ == "__main__":
    btokenizer = BasicTokenizer()
    tokenize_func = btokenizer.tokenize
    print(tokenize_func("关键阿斯顿词数量adsa"))