#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import infinity.rag_tokenizer
from common import settings
import logging

# Korean tokenizer import
_korean_tokenizer = None

def get_korean_tokenizer():
    """Get or create Korean tokenizer singleton instance"""
    global _korean_tokenizer
    if _korean_tokenizer is None:
        try:
            from .korean_tokenizer import KoreanTokenizer
            _korean_tokenizer = KoreanTokenizer()
            logging.info("[KOREAN_TOKENIZER] Korean tokenizer initialized successfully")
        except Exception as e:
            logging.warning(f"[KOREAN_TOKENIZER] Failed to initialize Korean tokenizer: {e}")
            _korean_tokenizer = False  # Mark as failed to avoid repeated attempts
    return _korean_tokenizer if _korean_tokenizer else None

class RagTokenizer(infinity.rag_tokenizer.RagTokenizer):
    def tokenize(self, line: str, force=False) -> str:
        if settings.DOC_ENGINE_INFINITY:
            return line
        else:
            # Korean tokenization check
            if not force and is_korean(line):
                kor_tokenizer = get_korean_tokenizer()
                if kor_tokenizer:
                    try:
                        tokens = kor_tokenizer.tokenize(line)
                        result = " ".join(tokens)
                        logging.debug("[KOREAN_TKS] {}".format(result))
                        return result
                    except Exception as e:
                        logging.warning(f"[KOREAN_TOKENIZER] Error: {e}, falling back to default")
                        # Fall through to default
            
            return super().tokenize(line)

    def fine_grained_tokenize(self, tks: str) -> str:
        if settings.DOC_ENGINE_INFINITY:
            return tks
        else:
            # For Korean text, return the same tokens
            if isinstance(tks, str) and is_korean(tks):
                logging.debug("[KOREAN_FINE_GRAINED] Using same tokens as base tokenization")
                return tks
            
            return super().fine_grained_tokenize(tks)

# class RagTokenizer(infinity.rag_tokenizer.RagTokenizer):

#     def tokenize(self, line: str) -> str:
#         if settings.DOC_ENGINE_INFINITY:
#             return line
#         else:
#             return super().tokenize(line)

#     def fine_grained_tokenize(self, tks: str) -> str:
#         if settings.DOC_ENGINE_INFINITY:
#             return tks
#         else:
#             return super().fine_grained_tokenize(tks)

def is_korean(text):
    """
    Check if the text is primarily Korean.
    Returns True if Korean characters (Hangul) make up more than 30% of the text.
    """
    if not text or len(text.strip()) == 0:
        return False
    
    korean_chars = sum(1 for c in text if '\uAC00' <= c <= '\uD7A3')  # Hangul Syllables
    total_chars = sum(1 for c in text if c.strip())
    
    if total_chars == 0:
        return False
    
    return korean_chars / total_chars > 0.3

def is_chinese(s):
    return infinity.rag_tokenizer.is_chinese(s)


def is_number(s):
    return infinity.rag_tokenizer.is_number(s)


def is_alphabet(s):
    return infinity.rag_tokenizer.is_alphabet(s)


def naive_qie(txt):
    return infinity.rag_tokenizer.naive_qie(txt)


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B
