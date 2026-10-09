from scripts.report_event_navigation import acceptance


def metrics(**differences):
    fields = ['cluster_safe_success', 'relaxed_safe_success', 'wall_contact_s',
              'obstacle_events_static', 'obstacle_events_dynamic', 'task_success']
    values = dict.fromkeys(fields, 0.)
    values['cluster_safe_success'] = .05
    values.update(differences)
    return {field: {'delta': value} for field, value in values.items()}


def test_gate_requires_both_safety_definitions_and_does_not_hide_contact_regression():
    assert acceptance(metrics())
    assert not acceptance(metrics(relaxed_safe_success=-.01))
    assert not acceptance(metrics(wall_contact_s=.01))
    assert not acceptance(metrics(obstacle_events_static=.01))
    assert not acceptance(metrics(obstacle_events_dynamic=.01))
    assert not acceptance(metrics(task_success=-.06))
    assert not acceptance(metrics(cluster_safe_success=0.))
