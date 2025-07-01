from fastapi import FastAPI, HTTPException
from typing import Dict
from scheduler import Task, forward_pass, backward_pass, critical_path

app = FastAPI(title="Project Scheduler")

# Example in-memory projects
data: Dict[str, Dict[str, Task]] = {
    "1": {
        "A": Task(id="A", duration=3),
        "B": Task(id="B", duration=2, dependencies=["A"]),
        "C": Task(id="C", duration=4, dependencies=["A"]),
        "D": Task(id="D", duration=2, dependencies=["B", "C"]),
    }
}

@app.get("/projects/{project_id}/critical-path")
def get_critical_path(project_id: str):
    if project_id not in data:
        raise HTTPException(status_code=404, detail="Project not found")

    tasks = data[project_id]
    forward_pass(tasks)
    backward_pass(tasks)
    path = critical_path(tasks)
    return {
        "critical_path": path,
        "tasks": {
            t.id: {
                "ES": t.ES,
                "EF": t.EF,
                "LS": t.LS,
                "LF": t.LF,
                "total_float": t.total_float,
            }
            for t in tasks.values()
        },
    }
