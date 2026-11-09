import os
import io
import time
import json
from typing import Optional
from PIL import Image
from tqdm import tqdm

from google import genai
from google.genai import types

from src.baseline_model import _encode_image_to_bytes, _save_json
from src.csg_schema import CSG_SCHEMA_INSTRUCTION, CSG_COT_PREAMBLE, compile_csg_to_cadquery, CSGTree
from src.visual_feedback import VisualFeedbackLoop
from src.code_repair import classify_csg_error


# Prompt template for Flash constrained repair (Class A errors only)
FLASH_REPAIR_PROMPT = """The following CSG JSON tree has a {error_class} error:

```json
{original_json}
```

Error: {error_message}

Fix ONLY the specific error. Do NOT change the topology or add/remove operations.
Rules:
- All dimensions must be positive and <= 200.0
- All translations must be in [-100.0, 100.0]
- All radius/diameter values must be positive
- Output the corrected JSON only, no explanation.
"""


def _safe_json_loads(txt: str) -> dict:
    """Parse JSON string safely with json-repair fallback for LLM formatting anomalies."""
    if not txt or not txt.strip():
        return {}
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    try:
        import json_repair
        data = json_repair.loads(txt)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


class EnhancedRAGGenerator:
    """CSG JSON Generation with Hierarchical Dual-Model Architecture.
    
    Uses a primary model (e.g. Gemini Pro) for spatial reasoning and CSG
    tree generation, and a repair model (e.g. Gemini Flash) for constrained
    parameter-only fixes on Class A errors. Class B errors (topology/geometry)
    trigger full regeneration instead of repair.
    """

    def __init__(
        self,
        retriever=None,
        model: str = "gemini-3.1-pro-preview",
        repair_model: str = "gemini-2.5-flash",
        n_shots: int = 0,
        use_cot: bool = True,
        include_reference: bool = False,
        enable_repair: bool = True,
        repair_max_attempts: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        api_key: Optional[str] = None,
        # Legacy parameter names for backward compatibility
        generator_model: Optional[str] = None,
    ):
        self.generator_model_id = generator_model or model
        self.repair_model_id = repair_model
        self.retriever = retriever
        self.n_shots = n_shots
        self.use_cot = use_cot
        self.enable_repair = enable_repair
        self.repair_max_attempts = repair_max_attempts
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.feedback_loop = VisualFeedbackLoop()

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("No API key found.")
        self.client = genai.Client(api_key=key)

        # Build system prompt with optional CoT preamble
        if use_cot:
            self.system_prompt = CSG_COT_PREAMBLE + "\n" + CSG_SCHEMA_INSTRUCTION
        else:
            self.system_prompt = CSG_SCHEMA_INSTRUCTION

    def _generate_csg(
        self,
        image: Image.Image,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Generate a CSG JSON tree from an image using the specified model.
        
        Args:
            image: Target image to generate CAD for
            model: Model ID override (defaults to self.generator_model_id)
            temperature: Temperature override
            
        Returns:
            Parsed CSG tree dict
        """
        model_id = model or self.generator_model_id
        temp = temperature if temperature is not None else self.temperature
        img_bytes = _encode_image_to_bytes(image)

        # Build prompt parts, optionally prepending RAG few-shot examples
        parts = []

        if self.retriever and self.n_shots > 0:
            try:
                similar_examples = self.retriever.retrieve(image)
                for i, ex in enumerate(similar_examples[:self.n_shots]):
                    ex_img_bytes = _encode_image_to_bytes(ex["image"])
                    parts.append(types.Part.from_text(
                        text=f"Example {i+1} — a similar 3D object and its correct CadQuery code:"
                    ))
                    parts.append(types.Part.from_bytes(data=ex_img_bytes, mime_type="image/png"))
                    parts.append(types.Part.from_text(
                        text=f"```python\n{ex['cadquery']}\n```"
                    ))
            except Exception as e:
                print(f"  [RAG] Retrieval failed, proceeding without few-shot: {e}")

        parts.append(types.Part.from_text(text="Generate a JSON CSG construction tree for this 3D object:"))
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        contents = [
            types.Content(role="user", parts=parts)
        ]

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=temp,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_schema=CSGTree
        )

        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=config,
                )
                txt = response.text or "{}"
                data = _safe_json_loads(txt)
                if data:
                    return data
                raise ValueError(f"Failed to parse CSG JSON: {txt[:100]}")
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  [Rate limit hit, sleeping 60s...]")
                    time.sleep(60)
                elif attempt == 4:
                    print(f"  Failed to generate CSG: {e}")
                    return {}
                else:
                    time.sleep(2)
        return {}

    def _validate_csg(self, csg_tree: dict) -> tuple[Optional[str], str]:
        """Validate a CSG tree by compiling and executing it.
        
        Returns:
            Tuple of (error_class, error_message).
            error_class is None if valid, "A" for repairable, "B" for unrecoverable.
        """
        if not csg_tree.get("steps"):
            return "B", "Empty CSG tree (no steps)"
        
        code = compile_csg_to_cadquery(csg_tree)
        
        # Try executing the compiled code in a subprocess
        from src.code_repair import _try_execute
        success, error_msg = _try_execute(code, from_csg_compiler=True)
        
        if success:
            return None, ""
        
        return classify_csg_error(error_msg), error_msg

    def _flash_repair(self, csg_tree: dict, error_msg: str) -> dict:
        """Use Flash model to repair a Class A error in the CSG tree.
        
        Only fixes JSON syntax, missing fields, and parameter bounds.
        Does NOT change topology or add/remove operations.
        """
        repair_prompt = FLASH_REPAIR_PROMPT.format(
            error_class="parameter/syntax",
            original_json=json.dumps(csg_tree, indent=2),
            error_message=error_msg,
        )

        config = types.GenerateContentConfig(
            system_instruction=CSG_SCHEMA_INSTRUCTION,
            temperature=0.1,  # Low temp for precise fixes
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_schema=CSGTree
        )

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.repair_model_id,
                    contents=[types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=repair_prompt)],
                    )],
                    config=config,
                )
                txt = response.text or "{}"
                data = _safe_json_loads(txt)
                if data:
                    return data
                return csg_tree
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"    [Flash Repair] Rate limit hit, sleeping 60s...")
                    time.sleep(60)
                elif attempt == 2:
                    print(f"    [Flash Repair] Failed: {e}")
                    return csg_tree
                else:
                    time.sleep(2)
        return csg_tree

    def _vlm_repair(self, csg_tree: dict, diagnosis: dict, target_image: Image.Image) -> dict:
        """Use VLM to repair broken JSON given step-by-step imagery (legacy)."""
        failed_idx = diagnosis.get("failed_step_index", 0)
        failed_op = diagnosis.get("failed_operation", "unknown")
        error_msg = diagnosis.get("error_message", "")
        last_good_image = diagnosis.get("last_good_image", None)

        repair_prompt = (
            f"You generated the following CSG JSON tree:\n"
            f"```json\n{json.dumps(csg_tree, indent=2)}\n```\n\n"
            f"The construction succeeded up to step {failed_idx}. "
            f"However, step {failed_idx} (operation: '{failed_op}') failed with this error:\n"
            f"'{error_msg}'\n\n"
            f"Please output a corrected JSON CSG tree that fixes this step so it generates the target object correctly."
        )

        parts = [types.Part.from_text(text=repair_prompt)]
        
        if last_good_image is not None:
            rendered_bytes = _encode_image_to_bytes(last_good_image)
            parts.insert(0, types.Part.from_bytes(data=rendered_bytes, mime_type="image/png"))
            parts.insert(1, types.Part.from_text(text="Here is a rendering of the 3D object right before it broke:"))
            
        img_bytes = _encode_image_to_bytes(target_image)
        parts.insert(0, types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        parts.insert(1, types.Part.from_text(text="This is the target image you are trying to match:"))

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.2,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_schema=CSGTree
        )

        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.repair_model_id,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                txt = response.text or "{}"
                data = _safe_json_loads(txt)
                if data:
                    return data
                return csg_tree
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"    [Repair] Rate limit hit, sleeping 60s...")
                    time.sleep(60)
                elif attempt == 4:
                    print(f"    VLM repair generation failed: {e}")
                    return csg_tree
                else:
                    time.sleep(2)
        return csg_tree

    def generate(self, image: Image.Image) -> str:
        """Generate CadQuery code via CSG JSON -> Compiler.
        
        Flow:
        1. Primary generation with Pro model (CoT + schema)
        2. Validate CSG tree (compile + execute)
        3. If Class A error: Flash constrained repair (max 2 attempts)
        4. If Class B error: Full regeneration with increased temperature
        5. Compile final CSG tree to CadQuery code
        """
        # Step 1: Primary generation
        csg_tree = self._generate_csg(image)

        # Step 2+3: Validation and repair loop
        if self.enable_repair and csg_tree.get("steps"):
            for attempt in range(self.repair_max_attempts):
                error_class, error_msg = self._validate_csg(csg_tree)
                
                if error_class is None:
                    # Valid — proceed
                    break
                elif error_class == "A":
                    # Class A: repairable by Flash
                    print(f"  [Class A error] {error_msg[:80]}... Sending to Flash repair.")
                    csg_tree = self._flash_repair(csg_tree, error_msg)
                elif error_class == "B":
                    # Class B: unrecoverable — regenerate with higher temperature
                    new_temp = self.temperature + 0.2 * (attempt + 1)
                    print(f"  [Class B error] {error_msg[:80]}... Regenerating (temp={new_temp:.1f}).")
                    csg_tree = self._generate_csg(
                        image,
                        temperature=min(new_temp, 1.0),
                    )

        # Step 4: Also run visual feedback repair if enabled and steps exist
        if self.enable_repair and csg_tree.get("steps"):
            for attempt in range(self.repair_max_attempts):
                step_results = self.feedback_loop.execute_stepwise(csg_tree)
                
                if all(r["success"] for r in step_results):
                    break
                    
                diagnosis = self.feedback_loop.diagnose_failure(step_results)
                error_msg = diagnosis.get("error_message", "")
                error_class = classify_csg_error(error_msg)
                
                if error_class == "A":
                    print(f"  [Visual Feedback] Step {diagnosis.get('failed_step_index')} failed (Class A). Flash repair...")
                    csg_tree = self._flash_repair(csg_tree, error_msg)
                else:
                    print(f"  [Visual Feedback] Step {diagnosis.get('failed_step_index')} failed (Class B). VLM repair...")
                    csg_tree = self._vlm_repair(csg_tree, diagnosis, image)

        # Step 5: Final compile
        return compile_csg_to_cadquery(csg_tree)

    def generate_batch(
        self,
        items: dict,
        delay: float = 1.5,
        save_path: Optional[str] = None,
        existing_predictions: Optional[dict] = None,
    ) -> dict:
        predictions = existing_predictions.copy() if existing_predictions else {}
        errors = {}
        ids = sorted(items.keys())

        for i, item_id in enumerate(tqdm(ids, desc="Generating (CSG)")):
            try:
                code = self.generate(items[item_id]["image"])
                predictions[item_id] = code
            except Exception as e:
                print(f"  Error generating {item_id}: {e}")
                errors[item_id] = str(e)
                predictions[item_id] = ""

            if delay > 0 and i < len(ids) - 1:
                time.sleep(delay)

            if save_path and (i + 1) % 10 == 0:
                _save_json(save_path, {"predictions": predictions, "errors": errors})

        if save_path:
            _save_json(save_path, {"predictions": predictions, "errors": errors})

        return predictions
