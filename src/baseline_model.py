"""
Baseline CadQuery code generator using Google Gemini VLM API.

Zero-shot and few-shot prompting for image → CadQuery code generation.
"""

import os
import re
import io
import base64
import time
import json
from typing import Optional
from PIL import Image
from tqdm import tqdm

from google import genai
from google.genai import types

from src.cadquery_reference import get_system_prompt


def _encode_image_to_bytes(image: Image.Image, max_size: int = 1024) -> bytes:
    """Encode a PIL Image to PNG bytes, resizing if needed."""
    # Resize if too large (save API tokens / speed)
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _clean_code_response(response_text: str) -> str:
    """Extract clean Python code from a VLM response.

    Handles markdown fencing, extra explanation text, etc.
    """
    text = response_text.strip()

    # Remove markdown code fences if present
    if "```" in text:
        # Try to extract from ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            text = matches[0].strip()
        else:
            # Fallback: remove all ``` lines
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

    # Remove any leading explanation lines (non-code)
    # Keep lines that look like Python code or comments
    lines = text.split("\n")
    code_started = False
    code_lines = []
    for line in lines:
        stripped = line.strip()
        # Heuristics for code vs explanation
        if not code_started:
            if (
                stripped.startswith("#")
                or stripped.startswith("import ")
                or stripped.startswith("from ")
                or "=" in stripped
                or stripped.startswith("result")
                or stripped.startswith("cq.")
                or stripped == ""
                or any(stripped.startswith(kw) for kw in [
                    "height", "width", "length", "depth", "radius",
                    "diameter", "thickness", "padding", "size", "offset",
                    "angle", "fillet", "chamfer", "hole",
                ])
            ):
                code_started = True
                code_lines.append(line)
        else:
            code_lines.append(line)

    if code_lines:
        text = "\n".join(code_lines).strip()

    # Remove `import cadquery as cq` if present (the metric's exec env already provides cq)
    text = re.sub(r"^import cadquery as cq\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^from cadquery import \*\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^import cadquery\s*\n?", "", text, flags=re.MULTILINE)

    return text.strip()


class BaselineGeminiGenerator:
    """Zero-shot or few-shot CadQuery code generation via Google Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        n_shots: int = 0,
        few_shot_examples: Optional[list[dict]] = None,
        use_cot: bool = False,
        include_reference: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            model: Gemini model name
            n_shots: Number of few-shot examples (0 = zero-shot)
            few_shot_examples: Pre-loaded few-shot examples [{image, cadquery}]
            use_cot: Use chain-of-thought prompting
            include_reference: Include CadQuery API reference in prompt
            temperature: Generation temperature
            max_tokens: Maximum output tokens
            api_key: Google API key (or set GOOGLE_API_KEY env var)
        """
        self.model_name = model
        self.n_shots = n_shots
        self.few_shot_examples = few_shot_examples or []
        self.use_cot = use_cot
        self.include_reference = include_reference
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialise Gemini client
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable, "
                "or pass api_key parameter."
            )
        self.client = genai.Client(api_key=key)

        self.system_prompt = get_system_prompt(
            use_cot=self.use_cot,
            include_reference=self.include_reference,
        )

    def _build_few_shot_contents(self) -> list:
        """Build few-shot example content parts for the message."""
        contents = []
        examples = self.few_shot_examples[: self.n_shots]

        for ex in examples:
            # User turn: image
            img_bytes = _encode_image_to_bytes(ex["image"])
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text="Generate CadQuery code for this 3D object:"),
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    ],
                )
            )
            # Model turn: code
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=ex["cadquery"])],
                )
            )

        return contents

    def generate(self, image: Image.Image) -> str:
        """Generate CadQuery code from a single image.

        Args:
            image: PIL Image of the CAD object

        Returns:
            Generated CadQuery code string
        """
        img_bytes = _encode_image_to_bytes(image)

        # Build contents
        contents = []

        # Add few-shot examples
        if self.n_shots > 0 and self.few_shot_examples:
            contents.extend(self._build_few_shot_contents())

        # Add the query image
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="Generate CadQuery code for this 3D object:"),
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                ],
            )
        )

        # Call Gemini API
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                return _clean_code_response(response.text or "")
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  [Rate limit hit, sleeping 60s...]")
                    time.sleep(60)
                elif attempt == 4:
                    raise e
                else:
                    time.sleep(2)
        return ""

    def generate_batch(
        self,
        items: dict,
        delay: float = 1.0,
        save_path: Optional[str] = None,
        existing_predictions: Optional[dict] = None,
    ) -> dict:
        """Generate CadQuery code for a batch of items.

        Args:
            items: Dict of {id: {"image": PIL.Image, ...}}
            delay: Delay between API calls (rate limiting)
            save_path: Optional path to save intermediate results
            existing_predictions: Optional dict of already completed predictions to merge with

        Returns:
            Dict of {id: generated_code_string}
        """
        predictions = existing_predictions.copy() if existing_predictions else {}
        errors = {}
        ids = sorted(items.keys())

        for i, item_id in enumerate(tqdm(ids, desc="Generating")):
            try:
                code = self.generate(items[item_id]["image"])
                predictions[item_id] = code
            except Exception as e:
                print(f"  Error generating {item_id}: {e}")
                errors[item_id] = str(e)
                predictions[item_id] = ""  # Empty = will fail VSR

            # Rate limiting
            if delay > 0 and i < len(ids) - 1:
                time.sleep(delay)

            # Periodic save
            if save_path and (i + 1) % 10 == 0:
                _save_json(save_path, {"predictions": predictions, "errors": errors})

        # Final save
        if save_path:
            _save_json(save_path, {"predictions": predictions, "errors": errors})

        return predictions


def _save_json(path: str, data: dict):
    """Save data as JSON, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
