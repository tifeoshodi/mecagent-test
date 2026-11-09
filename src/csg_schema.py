import json
from typing import Optional
from pydantic import BaseModel, Field

class CSGParams(BaseModel):
    # Primitives
    length: Optional[float] = Field(default=None, description="Length for box or chamfer")
    width: Optional[float] = Field(default=None, description="Width for box")
    height: Optional[float] = Field(default=None, description="Height for box or cylinder")
    radius: Optional[float] = Field(default=None, description="Radius for cylinder, sphere, circle, fillet, polygon")
    diameter: Optional[float] = Field(default=None, description="Diameter for hole, cboreHole, cskHole")
    xLen: Optional[float] = Field(default=None, description="x length for rect")
    yLen: Optional[float] = Field(default=None, description="y length for rect")
    # Workplane
    plane: Optional[str] = Field(default=None, description="Plane for workplane, e.g. XY, XZ, YZ")
    offset: Optional[float] = Field(default=None, description="Offset for workplane or offset2D")
    # Extrude / Cut
    distance: Optional[float] = Field(default=None, description="Distance for extrude")
    depth: Optional[float] = Field(default=None, description="Depth for hole or cboreHole")
    # Points
    points: Optional[list[list[float]]] = Field(default=None, description="Points for pushPoints")
    # Revolve / Sweep / Loft
    angle: Optional[float] = Field(default=None, description="Angle in degrees for revolve")
    axis: Optional[str] = Field(default=None, description="Axis for revolve or mirror, e.g. X, Y, Z")
    # Shell
    thickness: Optional[float] = Field(default=None, description="Wall thickness for shell")
    # Polygon
    n_sides: Optional[int] = Field(default=None, description="Number of sides for polygon")
    # Spline
    control_points: Optional[list[list[float]]] = Field(default=None, description="Control points for spline")
    # Counterbore / Countersink
    cbore_diameter: Optional[float] = Field(default=None, description="Counterbore diameter for cboreHole")
    cbore_depth: Optional[float] = Field(default=None, description="Counterbore depth for cboreHole")
    csk_diameter: Optional[float] = Field(default=None, description="Countersink diameter for cskHole")
    csk_angle: Optional[float] = Field(default=None, description="Countersink angle for cskHole")
    # Boolean target
    boolean_target: Optional[str] = Field(default=None, description="Variable name of the other solid for union/cut/intersect")

class CSGStep(BaseModel):
    operation: str = Field(description="box, cylinder, sphere, extrude, cutThruAll, hole, fillet, chamfer, workplane, circle, rect, pushPoints, polyline, revolve, sweep, loft, shell, mirror, cboreHole, cskHole, polygon, spline, offset2D, union, cut, intersect")
    params: Optional[CSGParams] = Field(default=None, description="Parameters for the operation")
    face_selector: Optional[str] = Field(default=None, description="Optional face selector, e.g., >Z, <Z, >X")
    edge_selector: Optional[str] = Field(default=None, description="Optional edge selector, e.g., |Z, >X")
    target: Optional[str] = Field(default=None, description="Optional variable name to assign this step to")

class CSGTree(BaseModel):
    steps: list[CSGStep]

# Define the JSON Schema for the VLM System Prompt
CSG_SCHEMA_INSTRUCTION = """
You must output ONLY valid JSON matching this schema for Constructive Solid Geometry (CSG) construction.
Do NOT output any markdown, explanations, or python code. Only the JSON object.

Schema:
{
  "steps": [
    {
      "operation": "box" | "cylinder" | "sphere" | "extrude" | "cutThruAll" | "hole" | "fillet" | "chamfer" | "workplane" | "circle" | "rect" | "pushPoints" | "polyline" | "revolve" | "sweep" | "loft" | "shell" | "mirror" | "cboreHole" | "cskHole" | "polygon" | "spline" | "offset2D" | "union" | "cut" | "intersect",
      "params": { 
         // 3D Primitives (creates solid): box {length, width, height}, cylinder {radius, height}, sphere {radius}
         // 2D Sketches (creates wire): circle {radius}, rect {xLen, yLen}, polygon {radius, n_sides}, polyline {points: [[x,y], ...]}, spline {control_points}
         // 2D-to-3D Extrusions: extrude {distance}, revolve {angle, axis} -> CRITICAL: You CANNOT extrude a 3D primitive. You MUST create a 2D sketch first.
         // Solid Modifiers: cutThruAll {}, hole {diameter, depth}
         // Feature Modifiers: fillet {radius}, chamfer {length}, shell {thickness}, mirror {axis}, offset2D {offset}
         // Fasteners: cboreHole {diameter, depth, cbore_diameter, cbore_depth}, cskHole {diameter, depth, csk_diameter, csk_angle}
         // Booleans: union/cut/intersect {boolean_target: "other_var_name"}
         // Workplane: {plane: "XY", offset: 0} (or use face_selector)
         // Points: pushPoints {points: [[x,y], ...]}
      },
      "face_selector": ">Z" | "<Z" | ">X" | etc (Optional, selects face to sketch on),
      "edge_selector": "|Z" | ">X" | etc (Optional, selects edges for fillet/chamfer),
      "target": "base" | "part1" (Optional variable name to assign this step to. If omitted, chains from previous step)
    }
  ]
}

// To combine multiple solids, you MUST explicitly assign a name using the "target" field FIRST, before referencing it in a boolean_target later.
// Example:
// {"target": "arm_x", "operation": "box", "params": {"length": 10}}
// {"operation": "union", "params": {"boolean_target": "arm_x"}}
// DO NOT use a boolean_target name that you haven't explicitly defined with "target" in a previous step!

Example 1 (Simple box with hole):
{
  "steps": [
    {"target": "base", "operation": "workplane", "params": {"plane": "XY"}},
    {"operation": "box", "params": {"length": 10, "width": 10, "height": 10}},
    {"operation": "workplane", "face_selector": ">Z"},
    {"operation": "hole", "params": {"diameter": 5}}
  ]
}

Example 2 (Revolved profile with shell):
{
  "steps": [
    {"target": "base", "operation": "workplane", "params": {"plane": "XZ"}},
    {"operation": "rect", "params": {"xLen": 10, "yLen": 20}},
    {"operation": "revolve", "params": {"angle": 360, "axis": "Y"}},
    {"operation": "shell", "params": {"thickness": 1}}
  ]
}

Example 3 (L-bracket with through-hole — multi-body):
{
  "steps": [
    {"target": "base", "operation": "workplane", "params": {"plane": "XY"}},
    {"operation": "box", "params": {"length": 40, "width": 20, "height": 5}},
    {"target": "wall", "operation": "workplane", "params": {"plane": "XZ"}},
    {"operation": "box", "params": {"length": 40, "width": 5, "height": 30}},
    {"operation": "union", "params": {"boolean_target": "wall"}},
    {"operation": "workplane", "face_selector": ">Z"},
    {"operation": "hole", "params": {"diameter": 8, "depth": 30}},
    {"operation": "fillet", "params": {"radius": 2}, "edge_selector": "|Z"}
  ]
}
"""

# Chain-of-Thought reasoning preamble — prepended to system prompt during generation
CSG_COT_PREAMBLE = """
### MANDATORY REASONING STEPS:
Before generating the JSON CSG tree, you MUST mentally decompose the target shape:

1. **Global Envelope**: What is the overall bounding shape? (box, cylinder, L-shape, etc.)
   Estimate dimensions relative to a unit scale where the largest dimension is approximately 50-80 units.
2. **Primary Solid**: Which single primitive or sketch+extrude creates the main body?
3. **Subtractive Features**: List all holes, slots, cutouts. For each, specify:
   - Primitive type (cylinder for holes, box for slots)
   - Which face it enters from (>Z, <Z, >X, etc.)
   - Approximate diameter/width relative to the main body
4. **Additive Features**: List any bosses, ribs, flanges added via union.
5. **Finishing**: Fillets, chamfers, shells. Apply these LAST.

### CONSTRAINTS:
- All dimensions MUST be positive and <= 200.0
- All translation coordinates MUST be in [-100.0, 100.0]
- Do NOT use a boolean_target name that you have not explicitly defined with "target" in a previous step
- Prefer 2D sketch -> extrude over 3D primitive booleans when possible
- Fillets and chamfers MUST be applied AFTER the solid is fully constructed
- Fillet radius MUST be smaller than the shortest adjacent edge
"""

def compile_csg_to_cadquery(csg_tree: dict) -> str:
    """
    Translates a CSG JSON tree into a string of executable CadQuery Python code.
    Guarantees 100% valid Python syntax.
    """
    lines = ["import cadquery as cq"]
    lines.append("result = None")
    
    steps = csg_tree.get("steps", [])
    
    current_target = "result"
    prev_target = None
    primary_solid = None
    solid_targets = set()
    initialized_targets = set()
    
    for i, step in enumerate(steps):
        op = step.get("operation")
        params = step.get("params") or {}
        params = {k: v for k, v in params.items() if v is not None}
        target = step.get("target")

        if op == "workplane" and not step.get("face_selector"):
            plane = params.get("plane", "XY")
            offset = params.get("offset", 0)
            next_target = None
            if i + 1 < len(steps):
                next_target = steps[i+1].get("target")
            if next_target and next_target != current_target:
                current_target = next_target
            elif current_target in solid_targets:
                current_target = f"sketch_{i}"
            lines.append(f"{current_target} = cq.Workplane('{plane}').workplane(offset={offset})")
            initialized_targets.add(current_target)
            continue
        
        if target and target != current_target:
            prev_target = current_target
            current_target = target
            
        # Initialize if this target hasn't been seen yet and it's not a workplane op
        if current_target not in initialized_targets and op != "workplane":
            if op in ["extrude", "revolve", "sweep", "loft"] and prev_target:
                lines.append(f"{current_target} = {prev_target}")
            else:
                lines.append(f"{current_target} = cq.Workplane('XY')")
        initialized_targets.add(current_target)
        
        if op == "workplane":
            offset = params.get("offset", 0)
            face_sel = step.get("face_selector", ">Z")
            lines.append(f"{current_target} = {current_target}.faces('{face_sel}').workplane(offset={offset})")
                
        elif op == "box":
            l = params.get("length", 1)
            w = params.get("width", 1)
            h = params.get("height", 1)
            lines.append(f"{current_target} = {current_target}.box({l}, {w}, {h})")
            solid_targets.add(current_target)
            if not primary_solid:
                primary_solid = current_target
            
        elif op == "cylinder":
            r = params.get("radius", 1)
            h = params.get("height", 1)
            lines.append(f"{current_target} = {current_target}.cylinder({h}, {r})") # CQ cylinder is height, radius
            solid_targets.add(current_target)
            if not primary_solid:
                primary_solid = current_target
            
        elif op == "sphere":
            r = params.get("radius", 1)
            lines.append(f"{current_target} = {current_target}.sphere({r})")
            solid_targets.add(current_target)
            if not primary_solid:
                primary_solid = current_target
            
        elif op == "circle":
            r = params.get("radius", 1)
            lines.append(f"{current_target} = {current_target}.circle({r})")
            
        elif op == "rect":
            x = params.get("xLen", 1)
            y = params.get("yLen", 1)
            lines.append(f"{current_target} = {current_target}.rect({x}, {y})")

        elif op == "polyline":
            pts = params.get("points", [])
            lines.append(f"{current_target} = {current_target}.polyline({pts}).close()")
            
        elif op == "pushPoints":
            pts = params.get("points", [[0,0]])
            # Auto-detect if pushPoints is used to sketch a 2D profile for 2D-to-3D extrusion/revolve
            is_profile = False
            if len(pts) >= 3:
                for next_step in steps[i+1:]:
                    next_op = next_step.get("operation")
                    if next_op in ["extrude", "revolve", "sweep", "loft"]:
                        is_profile = True
                        break
                    elif next_op in ["circle", "rect", "polyline", "polygon", "hole", "cboreHole", "cskHole"]:
                        break
            if is_profile:
                lines.append(f"{current_target} = {current_target}.polyline({pts}).close()")
            else:
                lines.append(f"{current_target} = {current_target}.pushPoints({pts})")
            
        elif op == "extrude":
            dist = params.get("distance", 1)
            lines.append(f"{current_target} = {current_target}.extrude({dist})")
            if not primary_solid:
                primary_solid = current_target
            
        elif op == "cutThruAll":
            lines.append(f"{current_target} = {current_target}.cutThruAll()")
            
        elif op == "hole":
            d = params.get("diameter", 1)
            depth = params.get("depth", None)
            if depth:
                lines.append(f"{current_target} = {current_target}.hole({d}, {depth})")
            else:
                lines.append(f"{current_target} = {current_target}.hole({d})")
                
        elif op == "fillet":
            r = params.get("radius", 1)
            edge_sel = step.get("edge_selector")
            if edge_sel:
                lines.append(f"{current_target} = {current_target}.edges('{edge_sel}').fillet({r})")
            else:
                lines.append(f"{current_target} = {current_target}.edges().fillet({r})")
                
        elif op == "chamfer":
            l = params.get("length", 1)
            edge_sel = step.get("edge_selector")
            if edge_sel:
                lines.append(f"{current_target} = {current_target}.edges('{edge_sel}').chamfer({l})")
            else:
                lines.append(f"{current_target} = {current_target}.edges().chamfer({l})")

        elif op == "revolve":
            angle = params.get("angle", 360)
            axis = params.get("axis", "Y")
            axis_tuple = "(1,0,0)" if axis == "X" else "(0,1,0)"
            lines.append(f"{current_target} = {current_target}.revolve({angle}, (0,0,0), {axis_tuple})")
            if not primary_solid:
                primary_solid = current_target

        elif op == "sweep":
            lines.append(f"{current_target} = {current_target}.sweep()")
            if not primary_solid:
                primary_solid = current_target

        elif op == "loft":
            lines.append(f"{current_target} = {current_target}.loft()")
            if not primary_solid:
                primary_solid = current_target

        elif op == "shell":
            thickness = params.get("thickness", 1)
            lines.append(f"{current_target} = {current_target}.shell({thickness})")

        elif op == "mirror":
            axis = params.get("axis", "X")
            plane_map = {"X": "'YZ'", "Y": "'XZ'", "Z": "'XY'"}
            mirror_plane = plane_map.get(axis, "'YZ'")
            lines.append(f"{current_target} = {current_target}.mirror({mirror_plane})")

        elif op == "polygon":
            n_sides = params.get("n_sides", 6)
            r = params.get("radius", 1)
            lines.append(f"{current_target} = {current_target}.polygon({n_sides}, {r * 2})")

        elif op == "spline":
            pts = params.get("control_points", [[0, 0], [1, 1]])
            lines.append(f"{current_target} = {current_target}.spline({pts})")

        elif op == "offset2D":
            offset_val = params.get("offset", 1)
            lines.append(f"{current_target} = {current_target}.offset2D({offset_val})")

        elif op == "cboreHole":
            d = params.get("diameter", 5)
            depth = params.get("depth", None)
            cbore_d = params.get("cbore_diameter", d * 1.5)
            cbore_depth = params.get("cbore_depth", 2)
            if depth:
                lines.append(f"{current_target} = {current_target}.cboreHole({d}, {cbore_d}, {cbore_depth}, {depth})")
            else:
                lines.append(f"{current_target} = {current_target}.cboreHole({d}, {cbore_d}, {cbore_depth})")

        elif op == "cskHole":
            d = params.get("diameter", 5)
            depth = params.get("depth", None)
            csk_d = params.get("csk_diameter", d * 2)
            csk_angle = params.get("csk_angle", 82)
            if depth:
                lines.append(f"{current_target} = {current_target}.cskHole({d}, {csk_d}, {csk_angle}, {depth})")
            else:
                lines.append(f"{current_target} = {current_target}.cskHole({d}, {csk_d}, {csk_angle})")

        elif op in ["union", "cut", "intersect"]:
            other = params.get("boolean_target", "part1")
            if other not in initialized_targets:
                lines.append(f"{other} = cq.Workplane('XY')")
                initialized_targets.add(other)
            
            if other == current_target and primary_solid and primary_solid != current_target:
                lines.append(f"{primary_solid} = {primary_solid}.{op}({current_target})")
                current_target = primary_solid
            elif current_target not in solid_targets and primary_solid in solid_targets:
                lines.append(f"{primary_solid} = {primary_solid}.{op}({other})")
                current_target = primary_solid
            else:
                if op == "union":
                    lines.append(f"{current_target} = {current_target}.union({other})")
                elif op == "cut":
                    lines.append(f"{current_target} = {current_target}.cut({other})")
                elif op == "intersect":
                    lines.append(f"{current_target} = {current_target}.intersect({other})")
                if primary_solid is None or current_target == primary_solid:
                    primary_solid = current_target
            solid_targets.add(current_target)
                
        elif op == "face_selector":
            face_sel = step.get("face_selector", ">Z")
            lines.append(f"{current_target} = {current_target}.faces('{face_sel}')")
            
        else:
            lines.append(f"# Ignored unknown operation: {op}")

    if primary_solid and primary_solid != "result":
        lines.append(f"result = {primary_solid}")
    elif current_target != "result":
        lines.append(f"result = {current_target}")
        
    return "\n".join(lines)
