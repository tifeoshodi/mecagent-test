from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Task:
    id: str
    duration: int
    dependencies: List[str] = field(default_factory=list)
    ES: int = 0  # Early start
    EF: int = 0  # Early finish
    LS: int = 0  # Late start
    LF: int = 0  # Late finish
    total_float: int = 0


def forward_pass(tasks: Dict[str, 'Task']):
    for task in tasks.values():
        if task.dependencies:
            task.ES = max(tasks[dep].EF for dep in task.dependencies)
        else:
            task.ES = 0
        task.EF = task.ES + task.duration


def backward_pass(tasks: Dict[str, 'Task']):
    max_finish = max(task.EF for task in tasks.values())
    # Start from tasks sorted in reverse of EF
    for task in sorted(tasks.values(), key=lambda t: t.EF, reverse=True):
        if not any(task.id in tasks[d].dependencies for d in tasks):
            task.LF = max_finish
        else:
            successors = [t for t in tasks.values() if task.id in t.dependencies]
            task.LF = min(s.LS for s in successors)
        task.LS = task.LF - task.duration
        task.total_float = task.LF - task.EF


def critical_path(tasks: Dict[str, 'Task']):
    return [t.id for t in tasks.values() if t.total_float == 0]
