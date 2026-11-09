import sys
from src.csg_schema import compile_csg_to_cadquery
from src.visual_feedback import VisualFeedbackLoop

def test_compiler():
    csg_tree = {
        "steps": [
            {"target": "base", "operation": "workplane", "params": {"plane": "XY"}},
            {"operation": "box", "params": {"length": 10, "width": 10, "height": 10}, "target": "base"},
            {"operation": "workplane", "face_selector": ">Z"},
            {"operation": "hole", "params": {"diameter": 5}}
        ]
    }
    
    code = compile_csg_to_cadquery(csg_tree)
    print("--- Compiled Code ---")
    print(code)
    print("---------------------\n")
    
    print("--- Running Visual Feedback Loop ---")
    feedback = VisualFeedbackLoop()
    results = feedback.execute_stepwise(csg_tree)
    
    for r in results:
        status = "SUCCESS" if r["success"] else f"FAILED: {r['error']}"
        img_info = "Image Generated" if r["image"] is not None else "No Image"
        print(f"Step {r['step_index']} ({r['operation']}): {status} | {img_info}")

if __name__ == "__main__":
    test_compiler()
