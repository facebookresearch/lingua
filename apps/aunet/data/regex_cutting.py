import logging
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import regex as re

logger = logging.getLogger(__name__)

@lru_cache(maxsize=256)
def utf8_byte_length(c):
    # Precompute byte lengths for quick lookup
    codepoint = ord(c)
    if codepoint <= 0x7F:
        return 1
    elif codepoint <= 0x7FF:
        return 2
    elif codepoint <= 0xFFFF:
        return 3
    else:
        return 4

def get_utf8_byte_length(first_byte_int):
    # Check the number of leading 1 bits in the integer representation
    if first_byte_int & 0b10000000 == 0:
        return 1  # 1-byte character (0xxxxxxx)
    elif first_byte_int & 0b11100000 == 0b11000000:
        return 2  # 2-byte character (110xxxxx)
    elif first_byte_int & 0b11110000 == 0b11100000:
        return 3  # 3-byte character (1110xxxx)
    elif first_byte_int & 0b11111000 == 0b11110000:
        return 4  # 4-byte character (11110xxx)
    else:
        return 0  # Invalid UTF-8 leading byte

def shrink_to_valid_bytes(byte_tokens: List[int]):
        def where_to_cut(_byte_tokens, end: bool):
            cut_idx = 0
            if end:
                _byte_tokens = _byte_tokens[:-5:-1]
            else:
                _byte_tokens = _byte_tokens[:5]
            
            for b in _byte_tokens:
                if not (b >= 128 and b < 192):
                    break
                cut_idx += 1
            
            num_expected_bytes = get_utf8_byte_length(_byte_tokens[cut_idx] if end else _byte_tokens[0])
            if num_expected_bytes == cut_idx + 1:
                cut_idx = None
            elif end:
                cut_idx = - (cut_idx + 1)

            return cut_idx
        
        start, end = where_to_cut(byte_tokens, False), where_to_cut(byte_tokens, True)
        return byte_tokens[start:end], start, end
        

def map_codepoint_to_byte(text):
    """Return the offset after the character like nb byte for this character"""
    length = len(text)

    mapping = np.zeros(length, dtype=np.int32)
    byte_offset = -1

    for char_offset, char in enumerate(text):
        byte_length = utf8_byte_length(char)
        byte_offset += byte_length
        mapping[char_offset] = byte_offset

    return mapping

WORD_PUNCT_RE = (r'( ?\p{L}{1,16})|\p{N}{1,3}| ?([^\s\p{L}\p{N}]){1,3}+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+', None)
NWORD_PUNCT_RE = (r'\p{{N}}{{1,3}}|(?:[^\s\p{{L}}\p{{N}}]{{1,3}}[\r\n]*)|\s*[\r\n]', r'(\S+?\s){{{gs}}}')
PUNCT_RE = (r'\d{1,3}(?=(\d{3})+(?!\d))|\d{1,3}(?=\D|$)|([^\s\p{L}\p{N}]){1,3}+[\r\n]*|\s*[\r\n]', None)

@dataclass
class RegexArgs:
    strategy: Dict[str, str] = field(default_factory=dict)

class RegexPool:
    def __init__(self, args: RegexArgs):
        self.patterns: List[re.Pattern] = []
        self.strategy = args.strategy
        for i, strategy in enumerate(args.strategy):
            logger.info(f"Strategy {i}: {strategy}")
            if strategy.startswith("word"):
                gs = int(args.strategy[strategy].split('@')[0])
                pat = (re.compile(NWORD_PUNCT_RE[0].format()), re.compile(NWORD_PUNCT_RE[1].format(gs=gs)))
                self.patterns.append(pat)
            elif strategy.startswith("pretok"):
                self.patterns.append((re.compile(WORD_PUNCT_RE[0]), None))
            elif strategy.startswith("punct"):
                self.patterns.append((re.compile(PUNCT_RE[0]), None))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    def str_offset(self, text: str):
        offsets = []
        for strat, pat_str in zip(self.strategy, self.patterns):
            start_pat_str, end_pat_str = pat_str
            tmp_offsets = []
            
            if start_pat_str is not None:
                start_matches = start_pat_str.finditer(text, concurrent=True)
                tmp_offsets.extend([match.start() for match in start_matches])
            
            if end_pat_str is not None:
                args = [1]
                if strat == "word":
                    args = list(map(int, self.strategy[strat].split('@')[1].split('-')))

                end_pat_str = end_pat_str.finditer(text, concurrent=True)
                end_match = [match.end()-1 for n, match in enumerate(end_pat_str) if any(n%mod == 0 for mod in args)]
                tmp_offsets.extend(end_match)
            
            tmp_offsets.sort()
            offsets.append(tmp_offsets)

        if len(offsets) > 1:
            pool_level = deque()
            not_empty = [len(off) > 0 for off in offsets]
            _offsets = deque()
            while any(not_empty):
                _max = -1
                recorded_i = -1
                same_should_pop = []
                for i, off in enumerate(offsets):
                    if not_empty[i] and off[-1] == _max:
                        same_should_pop.append(recorded_i)
                        recorded_i = i
                    if not_empty[i] and off[-1] > _max:
                        _max = off[-1]
                        recorded_i = i
                pool_level.appendleft(recorded_i)
                _offsets.appendleft(offsets[recorded_i].pop())
                not_empty[recorded_i] = bool(offsets[recorded_i])
                for i in same_should_pop:
                    offsets[i].pop()
                    not_empty[i] = bool(offsets[i])
            offsets = _offsets
        else:
            offsets = offsets[0]
            pool_level = [0] * len(offsets)
        
        return offsets, pool_level
    
    def get_levels_mask_prefill(self, byte: List[List[int]], size: Optional[int] = None, force_first: bool = False) -> Any:
        self.prefill_byte = []
        levels_mask = []
        for b in byte:
            self.prefill_byte.append(deque(b, maxlen=size))
            level_mask = self.get_levels_mask(b)
            if force_first:
                level_mask[0] = max(level_mask)
            levels_mask.append(level_mask)
        levels_mask = sum(levels_mask, [])
        return levels_mask
    
    def get_levels_mask_gen(self, byte: List[int]) -> Any:
        assert hasattr(self, "prefill_byte"), "You should call get_levels_mask_prefill before get_levels_mask_gen"
        assert len(byte) == len(self.prefill_byte), "You should call get_levels_mask_prefill with the same size before get_levels_mask_gen"
        
        levels_mask = []
        for b, new_b in zip(self.prefill_byte, byte):
            b.append(new_b)
            levels_mask.append(self.get_levels_mask(list(b))[-1])
        return levels_mask

    def get_levels_mask(self, byte: List[int]) -> Any:
        adjusted_bytes, start, end = shrink_to_valid_bytes(byte)

        txt_sgm = []
        flag = False
        for idx, b in enumerate(adjusted_bytes):
            if b > 255:
                if flag:
                    txt_sgm.append(idx)
                    flag = False
            else:
                if not flag:
                    txt_sgm.append(idx)
                    flag = True
        if flag:
            txt_sgm.append(len(adjusted_bytes))

        try:
            txt_seq = [bytes(adjusted_bytes[s:e]).decode("utf-8") for s, e in zip(txt_sgm[:-1:2], txt_sgm[1::2])]
        except Exception as e:
            txt_seq = [bytes(adjusted_bytes[s:e]).decode("utf-8", errors="replace") for s, e in zip(txt_sgm[:-1:2], txt_sgm[1::2])]
            logger.warning("Failed to decode a byte segment with UTF-8. Falling back to replacement characters. Error: %s", e)

        levels_mask = list(map(self._get_levels_mask, txt_seq))
        _levels_mask = [0] * len(adjusted_bytes)
        for s, e, level_mask in zip(txt_sgm[:-1:2], txt_sgm[1::2], levels_mask):
            _levels_mask[s:e] = level_mask[:]
        levels_mask = _levels_mask
       
        if start is not None:
            levels_mask = [0]*start + levels_mask
        if end is not None:
            levels_mask = levels_mask + [0]*(end*-1) 

        return levels_mask
    
    def _get_levels_mask(self, text: str):
        offsets, pool_level = self.str_offset(text)
        mapping = map_codepoint_to_byte(text)
        return self.__get_levels_mask(offsets, pool_level, mapping)

    def __get_levels_mask(self, offsets: List[int], pool_level: List[int], mapping: List[int]):
        levels_mask = [0] * (mapping[-1]+1)
        for i, off in enumerate(offsets):
            levels_mask[mapping[off]] = pool_level[i] + 1
        return levels_mask