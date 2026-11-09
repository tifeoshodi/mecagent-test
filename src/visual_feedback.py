import sys
import os
import pyvista as pv
from PIL import Image
import subprocess

# Add project root to path for subprocess execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.csg_schema import compile_csg_to_cadquery

def render_stl_to_file(stl_path: str, out_path: str):
    """Render an STL file to a PNG file using headless pyvista.
    
    WARNING: This function calls os._exit(0) at the end to prevent VTK C++
    destructor heap corruption on Windows. It MUST only be called in a
    subprocess (via render_stl()). Calling it directly will terminate the
    entire Python process.
    """
    mesh = pv.read(stl_path)
    
    # Initialize plotter with off_screen for headless rendering
    plotter = pv.Plotter(off_screen=True, window_size=[512, 512])
    plotter.set_background("white")
    
    # Add mesh with a distinct color and smooth shading
    plotter.add_mesh(mesh, color="lightblue", smooth_shading=True)
    
    # Set isometric camera for a clear 3D perspective
    plotter.camera_position = 'iso'
    
    # Capture screenshot to file
    plotter.screenshot(out_path, transparent_background=False)
    plotter.close()
    
    # Prevent VTK C++ destructors from causing heap corruption on Windows
    os._exit(0)

def render_stl(stl_path: str) -> Image.Image:
    """Render an STL file to a PIL Image using isolated subprocess."""
    out_path = stl_path + ".png"
    
    script_path = os.path.abspath(__file__)
    res = subprocess.run([sys.executable, script_path, stl_path, out_path], capture_output=True, text=True)
    
    if res.returncode != 0:
        raise RuntimeError(f"Headless rendering failed: {res.stderr}")
        
    return Image.open(out_path).copy()


class VisualFeedbackLoop:
    """Execute CadQuery CSG trees step-by-step, rendering intermediate 3D states."""
    
    def execute_stepwise(self, csg_tree: dict) -> list[dict]:
        step_results = []
        steps = csg_tree.get("steps", [])
        
        for i in range(1, len(steps) + 1):
            partial_tree = {"steps": steps[:i]}
            code = compile_csg_to_cadquery(partial_tree)
            
            success, error, img = self._execute_and_render(code)
            
            step_results.append({
                "step_index": i - 1,
                "code_so_far": code,
                "success": success,
                "error": error,
                "image": img,
                "operation": steps[i-1].get("operation")
            })
            
            if not success:
                break
                
        return step_results

    def _execute_and_render(self, code: str) -> tuple[bool, str, Image.Image]:
        script = f"""
import sys
import os
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
                print("CADQUERY_ERROR: The code compiled successfully but generated a topologically invalid/non-manifold 3D object. Check your dimensions and constraints.", file=sys.stderr)
                os._exit(1)
            
            print("CADQUERY_SUCCESS")
            try:
                # only try to export if it's a Solid or Compound or has solids
                if isinstance(solid_val, (cq.Solid, cq.Compound)) or (isinstance(var_value, cq.Workplane) and var_value.solids().vals()):
                    cq.exporters.export(solid_val, "temp_step.stl")
            except Exception as export_e:
                print("Export Error: " + str(export_e), file=sys.stderr)
            os._exit(0)
    print("No CadQuery object (Workplane/Solid/Compound) found in executed code", file=sys.stderr)
    os._exit(1)
except Exception as e:
    print(str(e), file=sys.stderr)
    os._exit(1)
"""
        try:
            res = subprocess.run([sys.executable, "-c", script], input=code, capture_output=True, text=True, timeout=15)
            if "CADQUERY_SUCCESS" in res.stdout or res.returncode == 0:
                img = None
                if os.path.exists("temp_step.stl"):
                    img = render_stl("temp_step.stl")
                return True, "", img
            return False, res.stderr.strip(), None
        except subprocess.TimeoutExpired:
            return False, "TimeoutError: The geometry caused the CAD engine to hang.", None

    def diagnose_failure(self, step_results: list[dict]) -> dict:
        for i, res in enumerate(step_results):
            if not res["success"]:
                last_good_img = step_results[i-1]["image"] if i > 0 else None
                return {
                    "failed_step_index": res["step_index"],
                    "failed_operation": res["operation"],
                    "error_message": res["error"],
                    "last_good_image": last_good_img,
                }
        return {}

if __name__ == "__main__":
    if len(sys.argv) == 3:
        render_stl_to_file(sys.argv[1], sys.argv[2])
