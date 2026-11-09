"""
Code repair module for fixing common CadQuery code generation errors.

Provides rule-based and VLM-assisted repair strategies.
"""

import re
import textwrap
import sys
import subprocess
import os
from typing import Literal


def classify_csg_error(error_msg: str) -> Literal["A", "B"]:
    """Classify a CSG/CadQuery error into repairable (A) or unrecoverable (B).
    
    Class A (repairable by Flash):
        - JSON parse errors, missing fields, parameter out-of-bounds
    Class B (requires full regeneration):
        - Topology crashes, degenerate geometry, BRep failures
    
    Args:
        error_msg: The error message string from validation/execution
    
    Returns:
        "A" for repairable errors, "B" for unrecoverable errors
    """
    error_lower = error_msg.lower()
    
    # Class A: JSON / schema / parameter errors
    class_a_patterns = [
        "jsondecode", "json.decoder", "expecting value", "expecting ','",
        "unterminated string", "missing", "invalid literal",
        "keyerror", "key error", "field required",
        "must be positive", "must be in [", "out of bound",
        "unknown operation",
    ]
    for pattern in class_a_patterns:
        if pattern in error_lower:
            return "A"
    
    # Class B: Topology / geometry / BRep errors
    class_b_patterns = [
        "brep_api", "topods", "gp_pnt", "standard_constructionerror",
        "isvalid", "non-manifold", "topologically invalid",
        "zero", "degenerate", "volume", "selfinters",
        "cadquery_error", "timeout", "hang",
        "no cadquery object", "not a solid",
        "opencascade", "occ", "ocp",
    ]
    for pattern in class_b_patterns:
        if pattern in error_lower:
            return "B"
    
    # Default: treat unknown errors as Class B (safer to regenerate)
    return "B"


def _try_execute(code: str, from_csg_compiler: bool = False) -> tuple[bool, str]:
    """Try executing CadQuery code and return (success, error_message).
    
    Args:
        code: CadQuery Python code to execute
        from_csg_compiler: If True, allows .union()/.cut()/.intersect() 
            (CSG compiler legitimately emits these for multi-body objects)
    """
    cleaned = textwrap.dedent(code).strip()
    
    # Build banned tokens list based on source
    if from_csg_compiler:
        banned_tokens_str = '[".fuse("]'  # Only ban .fuse() for CSG path
    else:
        banned_tokens_str = '[".union(", ".intersect(", ".fuse("]'
    
    script = """
import sys
import os
import cadquery as cq
import numpy as np

code = sys.stdin.read()

# Sandbox Text Filtering
banned_tokens = BANNED_TOKENS_PLACEHOLDER
for token in banned_tokens:
    if token in code:
        print("CADQUERY_ERROR: Explicit 3D boolean operations like '" + token + "' are banned. You must construct your object using 2D Workplane sketches and extrusions to ensure topological validity.", file=sys.stderr)
        os._exit(1)

ns = {"cq": cq, "cadquery": cq, "np": np, "numpy": np, "__builtins__": __builtins__}
try:
    exec(code, ns)
    for var_name, var_value in ns.items():
        if isinstance(var_value, (cq.Workplane, cq.Solid, cq.Compound)):
            solid_val = var_value.val() if isinstance(var_value, cq.Workplane) else var_value
            if not solid_val.isValid():
                print("CADQUERY_ERROR: The code compiled successfully but generated a topologically invalid/non-manifold 3D object. Use clean 2D sketch extrusions instead of haphazardly intersecting 3D primitives.", file=sys.stderr)
                os._exit(1)
            try:
                bb = solid_val.BoundingBox()
                max_dim = max(bb.xlen, bb.ylen, bb.zlen)
                if max_dim > 1000:
                    print(f"CADQUERY_ERROR: The generated object is far too large (dimension {max_dim:.1f} > 1000). You are likely using massive primitive shapes to visually hack a cut. Scale down your dimensions and use proper sketching.", file=sys.stderr)
                    os._exit(1)
            except Exception as bb_e:
                print("CADQUERY_ERROR: Could not compute bounding box: " + str(bb_e), file=sys.stderr)
                os._exit(1)
                
            print("CADQUERY_SUCCESS")
            try:
                cq.exporters.export(var_value, "temp_success.stl")
            except Exception as export_e:
                print("Export Error: " + str(export_e), file=sys.stderr)
            os._exit(0)
    print("No CadQuery object (Workplane/Solid/Compound) found in executed code", file=sys.stderr)
    os._exit(1)
except Exception as e:
    print(str(e), file=sys.stderr)
    os._exit(1)
""".replace("BANNED_TOKENS_PLACEHOLDER", banned_tokens_str)
    try:
        res = subprocess.run(
            [sys.executable, "-c", script],
            input=cleaned,
            capture_output=True,
            text=True,
            timeout=25
        )
        if "CADQUERY_SUCCESS" in res.stdout or res.returncode == 0:
            return True, ""
        else:
            return False, res.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "TimeoutError: The generated geometry caused the CAD engine to hang. Please simplify the boolean operations or rely on 2D sketch extrusion instead of 3D booleans."


def _fix_indentation(code: str) -> str:
    """Fix common indentation issues."""
    return textwrap.dedent(code).strip()


def _fix_unclosed_parens(code: str) -> str:
    """Attempt to fix unclosed parentheses/brackets."""
    open_parens = code.count("(") - code.count(")")
    open_brackets = code.count("[") - code.count("]")
    open_braces = code.count("{") - code.count("}")

    if open_parens > 0:
        code = code.rstrip() + ")" * open_parens
    if open_brackets > 0:
        code = code.rstrip() + "]" * open_brackets
    if open_braces > 0:
        code = code.rstrip() + "}" * open_braces

    return code


def _fix_common_api_errors(code: str) -> str:
    """Fix common CadQuery API misuse patterns."""
    # Fix `cq.Workplane().box(...)` when `.box(...)` is called without workplane spec
    code = re.sub(r'cq\.Workplane\(\)\.', 'cq.Workplane("XY").', code)

    # Fix `Workplane` without `cq.` prefix
    code = re.sub(r'(?<!\.)(?<!cq\.)Workplane\(', 'cq.Workplane(', code)

    # Fix `.extrude()` called without argument (add default 10)
    code = re.sub(r'\.extrude\(\s*\)', '.extrude(10)', code)

    return code


def _remove_import_lines(code: str) -> str:
    """Remove import statements (exec namespace provides them)."""
    lines = code.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def rule_based_repair(code: str) -> str:
    """Apply rule-based fixes to CadQuery code.

    Args:
        code: Raw generated CadQuery code

    Returns:
        Repaired code string
    """
    code = _fix_indentation(code)
    code = _remove_import_lines(code)
    code = _fix_unclosed_parens(code)
    code = _fix_common_api_errors(code)
    return code


def _apply_visual_critique(code: str, vlm_repair_fn, image) -> tuple[str, bool]:
    """Apply visual critique if the code succeeded and generated an STL."""
    if not vlm_repair_fn or not os.path.exists("temp_success.stl"):
        return code, True
        
    try:
        from src.visual_feedback import render_stl
        rendered_img = render_stl("temp_success.stl")
        
        # Critique using the VLM
        print("  [Visual Critique] Rendering matched syntax, sending for visual refinement...")
        repaired = vlm_repair_fn(code, "", image, rendered_img)
        success, error = _try_execute(repaired)
        if success:
            return repaired, True
        else:
            print("  [Visual Critique] Refined code failed syntax/geometry, falling back to original code.")
            return code, True
    except Exception as e:
        print(f"  [Visual Critique] Failed: {e}")
        return code, True


def validate_and_repair(
    code: str,
    max_attempts: int = 2,
    vlm_repair_fn=None,
    image=None,
) -> tuple[str, bool]:
    """Validate code and attempt repair if it fails.

    Args:
        code: Generated CadQuery code
        max_attempts: Maximum repair attempts
        vlm_repair_fn: Optional function(code, error, image, rendered_img) -> repaired_code
            for VLM-assisted repair
        image: Original image for VLM repair context

    Returns:
        Tuple of (repaired_code, success_bool)
    """
    # First try: raw code
    success, error = _try_execute(code)
    if success:
        return _apply_visual_critique(code, vlm_repair_fn, image)

    # Attempt 1: rule-based repair
    repaired = rule_based_repair(code)
    success, error = _try_execute(repaired)
    if success:
        return _apply_visual_critique(repaired, vlm_repair_fn, image)

    # Attempt 2+: VLM-assisted repair
    if vlm_repair_fn is not None:
        for attempt in range(max_attempts - 1):
            try:
                repaired = vlm_repair_fn(repaired, error, image)
                success, error = _try_execute(repaired)
                if success:
                    return _apply_visual_critique(repaired, vlm_repair_fn, image)
            except Exception:
                break

    # All attempts failed — return the best effort
    return repaired, False
