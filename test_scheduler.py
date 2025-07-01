from scheduler import Task, forward_pass, backward_pass, critical_path

tasks = {
    'A': Task(id='A', duration=3),
    'B': Task(id='B', duration=2, dependencies=['A']),
    'C': Task(id='C', duration=4, dependencies=['A']),
    'D': Task(id='D', duration=2, dependencies=['B', 'C']),
}

forward_pass(tasks)
backward_pass(tasks)
path = critical_path(tasks)


def test_forward_backward_pass():
    assert tasks['A'].ES == 0 and tasks['A'].EF == 3
    assert tasks['D'].LF == tasks['D'].EF
    assert path == ['A', 'C', 'D']
