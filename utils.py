import re
from typing import List
from dataclasses import dataclass
import random

from transformers import AutoTokenizer, PreTrainedTokenizerBase, VisionEncoderDecoderModel

from metrics.valid_syntax_rate import evaluate_syntax_rate


SPECIAL_PATTERNS = {
    r'\.faces(">Z")': "[FACES_GT_Z]",
    r'\.cboreHole\(\)': "[CBOREHOLE]",
}


def merge_patterns(text: str) -> str:
    """Replace known patterns with placeholder tokens."""
    for pattern, token in SPECIAL_PATTERNS.items():
        text = re.sub(pattern, token, text)
    return text


def unmerge_patterns(text: str) -> str:
    for pattern, token in SPECIAL_PATTERNS.items():
        text = text.replace(token, pattern)
    return text


class CustomTokenizer:
    """Tokenizer wrapper that merges user defined token patterns."""

    def __init__(self, base_model: str = "gpt2") -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_PATTERNS.values())})

    def encode(self, text: str, **kwargs) -> List[int]:
        text = merge_patterns(text)
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, ids: List[int], **kwargs) -> str:
        text = self.tokenizer.decode(ids, **kwargs)
        return unmerge_patterns(text)


@dataclass
class SyntaxAwareGenerator:
    """Beam-search wrapper that only returns syntactically valid outputs."""

    model: VisionEncoderDecoderModel
    tokenizer: PreTrainedTokenizerBase
    num_beams: int = 5

    def generate(self, pixel_values, **generate_kwargs) -> str:
        outputs = self.model.generate(
            pixel_values=pixel_values,
            num_beams=self.num_beams,
            num_return_sequences=self.num_beams,
            **generate_kwargs,
        )
        sequences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for seq in sequences:
            result = evaluate_syntax_rate({"candidate": seq}, verbose=False)
            if result["successful"]:
                return seq
        return sequences[0]


def random_drop_token_spans(tokens: List[int], max_spans: int = 3, max_span_len: int = 3) -> List[int]:
    """Randomly drop 1-3 token spans from a token sequence."""
    if not tokens:
        return tokens
    num_spans = random.randint(1, max_spans)
    tokens = tokens.copy()
    for _ in range(num_spans):
        if not tokens:
            break
        start = random.randrange(len(tokens))
        span_len = random.randint(1, min(max_span_len, len(tokens) - start))
        del tokens[start : start + span_len]
    return tokens
