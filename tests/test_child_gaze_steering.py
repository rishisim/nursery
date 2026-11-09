"""The steering matrix changes only the authorized scientific choice."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'child_gaze_steering', Path(__file__).parents[1] / 'scripts/child_gaze_steering.py')
steering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(steering)


def test_steering_preserves_every_other_protocol_field():
    parent = {
        'conditions': ['video_only', 'video_gaze', 'video_shuffled_gaze', 'video_gaze_point'],
        'gaze': {'option': 'additive', 'dropout': .5, 'steering_sigma': .15, 'shuffle_seed': 0},
        'seeds': [0, 1, 2], 'epochs': 60, 'batch': 256,
        'picture_panels': {'picture_test': {'trials': 'frozen', 'cache': 'frozen'}},
        'selection': {'method': 'historical user design decision'},
    }
    saved = copy.deepcopy(parent)
    child = steering.steering_protocol(parent)
    assert parent == saved
    assert child['conditions'] == ['video_gaze', 'video_shuffled_gaze']
    assert child['gaze']['option'] == 'steering'
    child['conditions'] = saved['conditions']
    child['gaze']['option'] = 'additive'
    assert child == saved


def test_steering_rejects_unexpected_parent():
    with pytest.raises(ValueError, match='additive parent'):
        steering.steering_protocol({'gaze': {'option': 'steering'}})
