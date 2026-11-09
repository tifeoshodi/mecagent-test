import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.code_repair import validate_and_repair, _try_execute

print("Testing Syntax Error:")
code_syntax_error = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10\n"
success, err = _try_execute(code_syntax_error)
print(f"Success: {success}, Err: {err}")

print("\nTesting Infinite Loop Geometry:")
code_inf_loop = """import cadquery as cq
while True:
    pass
"""
success, err = _try_execute(code_inf_loop)
print(f"Success: {success}, Err: {err}")

print("\nTesting Valid Geometry:")
code_valid = """import cadquery as cq
result = cq.Workplane('XY').box(10, 10, 10)
"""
success, err = _try_execute(code_valid)
print(f"Success: {success}, Err: {err}")
