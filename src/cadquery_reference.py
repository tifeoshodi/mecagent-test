"""
CadQuery API reference snippets for injection into VLM prompts.
"""

CADQUERY_OPERATIONS_REFERENCE = """
## CadQuery API Quick Reference

### Workplane Creation
- `cq.Workplane("XY")` — Create workplane on XY, XZ, or YZ plane

### 3D Primitives
- `.box(length, width, height)` — Centered rectangular box
- `.sphere(radius)` — Sphere
- `.cylinder(height, radius)` — Cylinder

### 2D Sketching
- `.circle(radius)` — Circle on current workplane
- `.rect(width, height)` — Rectangle on current workplane
- `.polygon(nSides, diameter)` — Regular polygon
- `.ellipse(x_radius, y_radius)` — Ellipse
- `.slot2D(length, diameter)` — Slot shape
- `.polyline(points)` — Polyline from point list
- `.close()` — Close an open wire

### 3D Operations
- `.extrude(distance)` — Extrude sketch into solid
- `.revolve(angleDegrees)` — Revolve sketch around axis
- `.loft()` — Loft between sections
- `.sweep(path)` — Sweep along a path

### Hole Operations
- `.hole(diameter)` — Through hole at center
- `.cboreHole(holeDia, boreDia, boreDepth)` — Counterbore hole
- `.cskHole(holeDia, cskDia, cskAngle)` — Countersink hole

### Edge/Face Operations
- `.fillet(radius)` — Fillet selected edges
- `.chamfer(distance)` — Chamfer selected edges
- `.shell(thickness)` — Hollow shell (negative = inward)

### Selectors
- `.faces(">Z")` — Top face (highest Z)
- `.faces("<Z")` — Bottom face (lowest Z)
- `.faces(">X")`, `.faces("<X")` — Side faces
- `.edges("|Z")` — Edges parallel to Z axis
- `.edges(">Z")` — Top edges
- `.vertices()` — All vertices of current selection

### Workplane Navigation
- `.workplane()` — New workplane on selected face
- `.center(x, y)` — Move workplane center
- `.transformed(offset=(x,y,z), rotate=(rx,ry,rz))` — Transform workplane

### Boolean Operations
- `.union(other)` — Boolean union
- `.cut(other)` — Boolean subtraction
- `.intersect(other)` — Boolean intersection

### Patterns
- `.rect(w, h, forConstruction=True).vertices()` — Rectangular pattern
- `.polygon(n, d, forConstruction=True).vertices()` — Circular pattern
- `.rarray(xSpacing, ySpacing, xCount, yCount)` — Rectangular array
- `.pushPoints([(x1,y1), (x2,y2)])` — Position at specific points
""".strip()


SYSTEM_PROMPT_BASE = """You are an expert CadQuery Python programmer. Your task is to generate valid CadQuery Python code that creates the 3D CAD object shown in the image.

RULES:
1. Output ONLY valid Python code — no markdown fencing, no explanations, no comments about the image.
2. The code must use the `cq` module (already imported as `import cadquery as cq`).
3. Store the final 3D object in a variable named `result`.
4. Use meaningful variable names for dimensions.
5. The code must be directly executable without any additional imports.
6. Produce clean, minimal code that accurately reconstructs the geometry.
7. **STRICT 2D ENFORCEMENT:** You MUST build your object using ONLY 2D `cq.Workplane` sketching and `.extrude()`. You are FORBIDDEN from using explicit 3D boolean operators like `.union()`, `.intersect()`, or `.cut()` on raw solids, as they cause non-manifold geometry.
8. **TOPOLOGICAL REFERENCING:** NEVER use absolute 3D coordinates (like `.translate()`). You MUST use CadQuery selectors (e.g., `.faces(">Z").workplane()`) to position sketches relative to existing geometry.
9. **ASSERTIONS:** At the end of your script, write an `assert` statement to verify the properties of your `result` (e.g., `assert result.faces().size() > 0`).

{reference}
""".strip()


SYSTEM_PROMPT_COT = """You are an expert CadQuery Python programmer. Your task is to generate valid CadQuery Python code that creates the 3D CAD object shown in the image.

APPROACH:
1. First, BRIEFLY describe the geometric decomposition in 1-2 lines as a Python comment at the top.
2. Then write the CadQuery code.

RULES:
1. Output ONLY valid Python code — no markdown fencing, no explanations outside comments.
2. The code must use the `cq` module (already imported as `import cadquery as cq`).
3. Store the final 3D object in a variable named `result`.
4. Use meaningful variable names for dimensions.
5. The code must be directly executable without any additional imports.
6. Produce clean, minimal code that accurately reconstructs the geometry.
7. **STRICT 2D ENFORCEMENT:** You MUST build your object using ONLY 2D `cq.Workplane` sketching and `.extrude()`. You are FORBIDDEN from using explicit 3D boolean operators like `.union()`, `.intersect()`, or `.cut()` on raw solids, as they cause non-manifold geometry.
8. **TOPOLOGICAL REFERENCING:** NEVER use absolute 3D coordinates (like `.translate()`). You MUST use CadQuery selectors (e.g., `.faces(">Z").workplane()`) to position sketches relative to existing geometry.
9. **ASSERTIONS:** At the end of your script, write an `assert` statement to verify the properties of your `result` (e.g., `assert result.faces().size() > 0`).

{reference}
""".strip()


def get_system_prompt(use_cot: bool = False, include_reference: bool = True) -> str:
    """Build the system prompt for CadQuery code generation.

    Args:
        use_cot: If True, use chain-of-thought prompt
        include_reference: If True, include CadQuery API reference

    Returns:
        Formatted system prompt string
    """
    reference = CADQUERY_OPERATIONS_REFERENCE if include_reference else ""
    template = SYSTEM_PROMPT_COT if use_cot else SYSTEM_PROMPT_BASE
    return template.format(reference=reference)
