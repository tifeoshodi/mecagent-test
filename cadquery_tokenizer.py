import re
from typing import List

class CadQueryTokenizer:
    """Simple tokenizer for CadQuery DSL."""

    SPECIAL_TOKENS = [".box(", ".hole(", ".rect("]

    def __init__(self):
        pattern = "|".join(re.escape(t) for t in self.SPECIAL_TOKENS)
        self._special_re = re.compile(f"({pattern})")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text splitting on whitespace but keeping CadQuery calls intact."""
        # Surround special patterns with spaces
        text = self._special_re.sub(lambda m: f" {m.group(1)} ", text)
        # Split on whitespace
        tokens = text.split()
        return tokens


if __name__ == "__main__":
    sample = 'result = cq.Workplane("XY").box(10, 10, 10).faces(\">Z\").workplane().hole(5)'
    tok = CadQueryTokenizer()
    print(tok.tokenize(sample))
