"""
Monte Carlo Tree Search (MCTS) Generator for CAD Construction.

Treats CAD generation as a sequential decision problem:
- State: A partial CSG tree (list of steps completed so far)
- Action: Append one new CSGStep to the tree
- Reward: Scoring function evaluating partial constructions against target image

Uses UCB1 for selection, VLM for expansion, and CLIP similarity for evaluation.
"""

import os
import io
import sys
import math
import time
import json
import subprocess
import numpy as np
import requests
import base64
from typing import Optional
from PIL import Image
from copy import deepcopy

from google import genai
from google.genai import types

from src.baseline_model import _encode_image_to_bytes
from src.csg_schema import CSG_SCHEMA_INSTRUCTION, compile_csg_to_cadquery, CSGTree


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(self, state: dict, parent=None, action: dict = None):
        """
        Args:
            state: A CSG tree dict {"steps": [...]} representing partial construction.
            parent: Parent MCTSNode.
            action: The CSGStep dict that was appended to reach this node.
        """
        self.state = state  # {"steps": [...]}
        self.parent = parent
        self.action = action  # The step that led to this node
        self.children: list["MCTSNode"] = []
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.is_terminal: bool = False
        self.is_expanded: bool = False
        self._cached_score: Optional[float] = None

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb1(self, exploration_weight: float = 1.41) -> float:
        """Upper Confidence Bound for tree traversal."""
        if self.visits == 0:
            return float("inf")
        exploit = self.avg_reward
        explore = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploit + explore

    @property
    def depth(self) -> int:
        return len(self.state.get("steps", []))


# ---------------------------------------------------------------------------
# Scoring Function
# ---------------------------------------------------------------------------

def _execute_cad_code(code: str, timeout: int = 15) -> tuple[bool, Optional[str]]:
    """Execute CadQuery code in a subprocess. Returns (success, stl_path)."""
    stl_path = "temp_mcts.stl"
    script = f"""
import sys, os
import cadquery as cq
import numpy as np

code = sys.stdin.read()
ns = {{"cq": cq, "cadquery": cq, "np": np, "numpy": np, "__builtins__": __builtins__}}
try:
    exec(code, ns)
    for var_name, var_value in ns.items():
        if isinstance(var_value, (cq.Workplane, cq.Solid, cq.Compound)):
            solid_val = var_value.val() if isinstance(var_value, cq.Workplane) else var_value
            if hasattr(solid_val, 'isValid') and not solid_val.isValid():
                os._exit(1)
            try:
                if isinstance(solid_val, (cq.Solid, cq.Compound)) or (isinstance(var_value, cq.Workplane) and var_value.solids().vals()):
                    cq.exporters.export(solid_val, "{stl_path}")
            except:
                pass
            print("SUCCESS")
            os._exit(0)
    os._exit(1)
except Exception as e:
    print(str(e), file=sys.stderr)
    os._exit(1)
"""
    try:
        res = subprocess.run(
            [sys.executable, "-c", script],
            input=code,
            capture_output=True, text=True, timeout=timeout
        )
        if "SUCCESS" in res.stdout and os.path.exists(stl_path):
            return True, stl_path
        return False, None
    except subprocess.TimeoutExpired:
        return False, None


def compute_clip_similarity(
    image_a: Image.Image,
    image_b: Image.Image,
    embedder=None,
) -> float:
    """Compute CLIP cosine similarity between two images.
    
    If no embedder is provided, falls back to a basic structural similarity.
    """
    if embedder is not None:
        emb_a = embedder.embed_single(image_a)
        emb_b = embedder.embed_single(image_b)
        cos_sim = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
        )
        return float(cos_sim)

    # Fallback: simple pixel-level structural comparison
    a = np.array(image_a.resize((64, 64)).convert("L"), dtype=np.float32) / 255.0
    b = np.array(image_b.resize((64, 64)).convert("L"), dtype=np.float32) / 255.0
    mse = np.mean((a - b) ** 2)
    return float(1.0 - mse)


def render_stl_to_image(stl_path: str) -> Optional[Image.Image]:
    """Render an STL to a PIL Image using pyvista in subprocess."""
    from src.visual_feedback import render_stl
    try:
        return render_stl(stl_path)
    except Exception:
        return None


def score_partial_tree(
    partial_tree: dict,
    target_image: Image.Image,
    embedder=None,
    w_compile: float = 0.3,
    w_similarity: float = 0.5,
    w_complexity: float = 0.2,
    max_steps: int = 15,
) -> float:
    """Score a partial CSG tree against a target image.
    
    Returns a float in [0, 1] combining:
    - compile_success (w_compile): Does the partial tree compile without errors?
    - silhouette_match (w_similarity): CLIP cosine similarity between rendered
      partial object and target image
    - complexity_bonus (w_complexity): Reward trees that use more operations
      (prevents degenerate single-box solutions)
    """
    steps = partial_tree.get("steps", [])
    if not steps:
        return 0.0

    # 1. Compile score
    code = compile_csg_to_cadquery(partial_tree)
    success, stl_path = _execute_cad_code(code)
    compile_score = 1.0 if success else 0.0

    # 2. Similarity score
    similarity_score = 0.0
    if success and stl_path:
        rendered = render_stl_to_image(stl_path)
        if rendered is not None:
            similarity_score = compute_clip_similarity(
                rendered, target_image, embedder
            )
            # Normalize from [-1, 1] to [0, 1]
            similarity_score = (similarity_score + 1.0) / 2.0

    # 3. Complexity bonus (diminishing returns)
    n_steps = len(steps)
    complexity_score = min(n_steps / max_steps, 1.0)

    if compile_score == 0.0:
        return 0.0

    total = (
        w_compile * compile_score
        + w_similarity * similarity_score
        + w_complexity * complexity_score
    )
    return min(max(total, 0.0), 1.0)


# ---------------------------------------------------------------------------
# MCTS Generator
# ---------------------------------------------------------------------------

class MCTSGenerator:
    """Monte Carlo Tree Search generator for CAD construction."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        budget: int = 20,
        n_candidates: int = 3,
        max_depth: int = 12,
        exploration_weight: float = 1.41,
        temperature: float = 0.4,
        max_tokens: int = 8192,
        embedder=None,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
    ):
        """
        Args:
            model: Gemini model name or Ollama model name.
            budget: Total number of MCTS iterations (VLM calls).
            n_candidates: Number of candidate next-steps per expansion.
            max_depth: Maximum CSG tree depth before marking terminal.
            exploration_weight: UCB1 exploration constant.
            temperature: VLM sampling temperature (higher = more diverse candidates).
            max_tokens: Max output tokens for VLM.
            embedder: Optional CLIP embedder for similarity scoring.
            api_key: Gemini API key.
            api_provider: "gemini" or "ollama".
        """
        self.model_name = model
        self.budget = budget
        self.n_candidates = n_candidates
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedder = embedder
        self.api_provider = api_provider

        if self.api_provider == "gemini":
            key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError("No API key found.")
            self.client = genai.Client(api_key=key)
        else:
            self.client = None
            
        self.system_prompt = CSG_SCHEMA_INSTRUCTION

    # --- MCTS Core ---

    def select(self, node: MCTSNode) -> MCTSNode:
        """Walk tree using UCB1 to find the most promising unexpanded leaf."""
        while node.is_expanded and node.children:
            node = max(node.children, key=lambda c: c.ucb1(self.exploration_weight))
        return node

    def expand(self, node: MCTSNode, target_image: Image.Image) -> list[MCTSNode]:
        """Ask VLM to propose candidate next steps, create child nodes."""
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return []

        candidates = self._vlm_propose_steps(node.state, target_image)
        children = []
        for candidate_step in candidates:
            child_state = deepcopy(node.state)
            child_state["steps"].append(candidate_step)
            child = MCTSNode(state=child_state, parent=node, action=candidate_step)
            children.append(child)

        node.children = children
        node.is_expanded = True
        return children

    def evaluate(self, node: MCTSNode, target_image: Image.Image) -> float:
        """Score a node's partial construction against the target."""
        if node._cached_score is not None:
            return node._cached_score
        score = score_partial_tree(
            node.state, target_image, self.embedder
        )
        node._cached_score = score
        return score

    def backpropagate(self, node: MCTSNode, reward: float):
        """Update visit counts and rewards up the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def best_path(self, root: MCTSNode) -> dict:
        """Extract the highest-reward complete path from the tree."""
        best_node = root
        best_score = -1.0

        def _traverse(node: MCTSNode):
            nonlocal best_node, best_score
            score = node._cached_score if node._cached_score is not None else 0.0
            if score > best_score and node.depth > 0:
                best_score = score
                best_node = node
            for child in node.children:
                _traverse(child)

        _traverse(root)
        return best_node.state

    # --- Main Generate ---

    def generate(self, image: Image.Image, verbose: bool = False) -> str:
        """Run MCTS to generate CadQuery code for a target image.
        
        Args:
            image: Target 3D object image.
            verbose: Print progress info.
            
        Returns:
            CadQuery code string.
        """
        root = MCTSNode(state={"steps": []})

        for iteration in range(self.budget):
            if verbose:
                print(f"  [MCTS] Iteration {iteration + 1}/{self.budget}")

            # 1. SELECT
            leaf = self.select(root)

            # 2. EXPAND
            if leaf.is_terminal:
                # Re-evaluate terminal nodes
                reward = self.evaluate(leaf, image)
                self.backpropagate(leaf, reward)
                continue

            children = self.expand(leaf, image)
            if not children:
                leaf.is_terminal = True
                reward = self.evaluate(leaf, image)
                self.backpropagate(leaf, reward)
                continue

            # 3. EVALUATE each child
            best_child = None
            best_reward = -1.0
            for child in children:
                reward = self.evaluate(child, image)
                if reward > best_reward:
                    best_reward = reward
                    best_child = child

            # 4. BACKPROPAGATE from the best child
            if best_child:
                self.backpropagate(best_child, best_reward)

            # Rate limit protection
            time.sleep(1.0)

        # Extract best path and compile
        best_tree = self.best_path(root)
        code = compile_csg_to_cadquery(best_tree)

        if verbose:
            print(f"  [MCTS] Best tree has {len(best_tree.get('steps', []))} steps")
            print(f"  [MCTS] Code:\n{code}")

        return code

    def generate_batch(
        self,
        items: dict,
        delay: float = 1.5,
        save_path: Optional[str] = None,
        verbose: bool = False,
        existing_predictions: Optional[dict] = None,
    ) -> dict:
        """Generate predictions for a batch of items."""
        from tqdm import tqdm
        from src.baseline_model import _save_json

        predictions = existing_predictions.copy() if existing_predictions else {}
        errors = {}
        ids = sorted(items.keys())

        for i, item_id in enumerate(tqdm(ids, desc="Generating (MCTS)")):
            try:
                code = self.generate(items[item_id]["image"], verbose=verbose)
                predictions[item_id] = code
            except Exception as e:
                print(f"  Error generating {item_id}: {e}")
                errors[item_id] = str(e)
                predictions[item_id] = ""

            if delay > 0 and i < len(ids) - 1:
                time.sleep(delay)

            if save_path:
                _save_json(save_path, {"predictions": predictions, "errors": errors})

        if save_path:
            _save_json(save_path, {"predictions": predictions, "errors": errors})

        return predictions

    # --- VLM Interaction ---

    def _vlm_propose_steps(
        self, current_state: dict, target_image: Image.Image
    ) -> list[dict]:
        """Ask the VLM to propose candidate next CSG steps."""
        if self.api_provider == "ollama":
            return self._ollama_propose_steps(current_state, target_image)
        else:
            return self._gemini_propose_steps(current_state, target_image)

    def _ollama_propose_steps(
        self, current_state: dict, target_image: Image.Image
    ) -> list[dict]:
        steps_so_far = current_state.get("steps", [])
        n_steps = len(steps_so_far)

        if n_steps == 0:
            context = "The construction is empty. Propose the FIRST step (usually a workplane + base primitive)."
        else:
            tree_json = json.dumps(current_state, indent=2)
            context = (
                f"The construction so far has {n_steps} steps:\n"
                f"```json\n{tree_json}\n```\n\n"
                f"Propose the NEXT single step to add to this tree."
            )

        prompt = (
            f"{context}\n\n"
            f"You must output a COMPLETE CSG tree (with ALL existing steps plus your new step appended).\n"
            f"Do NOT remove or modify any existing steps. Only ADD one new step at the end.\n"
            f"The goal is to construct the 3D object shown in the target image."
        )

        buffered = io.BytesIO()
        target_image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        candidates = []
        for candidate_idx in range(self.n_candidates):
            for attempt in range(3):
                try:
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "system": self.system_prompt,
                        "images": [img_b64],
                        "format": "json",
                        "stream": False,
                        "options": {
                            "temperature": self.temperature
                        }
                    }
                    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
                    response.raise_for_status()
                    
                    data = response.json()
                    txt = data.get("response", "{}").strip()
                    
                    if txt.startswith("```json"):
                        txt = txt[7:]
                    elif txt.startswith("```"):
                        txt = txt[3:]
                    if txt.endswith("```"):
                        txt = txt[:-3]
                    txt = txt.strip()
                    
                    tree = json.loads(txt)
                    proposed_steps = tree.get("steps", [])

                    if len(proposed_steps) > n_steps:
                        new_step = proposed_steps[n_steps]
                        if new_step not in candidates:
                            candidates.append(new_step)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"    [MCTS] Ollama proposal failed: {e}")
                    else:
                        time.sleep(2)

            time.sleep(0.5)

        return candidates

    def _gemini_propose_steps(
        self, current_state: dict, target_image: Image.Image
    ) -> list[dict]:
        steps_so_far = current_state.get("steps", [])
        n_steps = len(steps_so_far)

        if n_steps == 0:
            context = "The construction is empty. Propose the FIRST step (usually a workplane + base primitive)."
        else:
            tree_json = json.dumps(current_state, indent=2)
            context = (
                f"The construction so far has {n_steps} steps:\n"
                f"```json\n{tree_json}\n```\n\n"
                f"Propose the NEXT single step to add to this tree."
            )

        prompt = (
            f"{context}\n\n"
            f"You must output a COMPLETE CSG tree (with ALL existing steps plus your new step appended).\n"
            f"Do NOT remove or modify any existing steps. Only ADD one new step at the end.\n"
            f"The goal is to construct the 3D object shown in the target image."
        )

        img_bytes = _encode_image_to_bytes(target_image)
        parts = [
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            types.Part.from_text(text="Target 3D object to construct:"),
            types.Part.from_text(text=prompt),
        ]

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_schema=CSGTree,
        )

        candidates = []
        for candidate_idx in range(self.n_candidates):
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[types.Content(role="user", parts=parts)],
                        config=config,
                    )
                    txt = response.text or "{}"
                    tree = json.loads(txt)
                    proposed_steps = tree.get("steps", [])

                    # Extract the newly proposed step(s)
                    if len(proposed_steps) > n_steps:
                        new_step = proposed_steps[n_steps]
                        # Avoid duplicate candidates
                        if new_step not in candidates:
                            candidates.append(new_step)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        time.sleep(30)
                    elif attempt == 2:
                        print(f"    [MCTS] VLM proposal failed: {e}")
                    else:
                        time.sleep(2)

            # Small delay between candidates
            time.sleep(0.5)

        return candidates
