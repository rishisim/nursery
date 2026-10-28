import json

import pandas as pd
import pytest

from nursery.learner.cross_situational import CATEGORIES, eligible_categories, sample_rows


def test_visibility_proxy_requires_size_and_central_cell():
    objects = [dict(category='big', visible_px=100, cells=[5]),
               dict(category='small', visible_px=99, cells=[5]),
               dict(category='edge', visible_px=500, cells=[0])]
    assert eligible_categories(json.dumps(objects)) == {'big'}


def test_balanced_sampling_counts_replays_across_held_status():
    rows = []
    for part, clips in [('train', 20), ('dev', 5), ('test', 5)]:
        for category in CATEGORIES:
            for clip in range(clips):
                for held in (True, False):
                    for i in range(4):
                        rows.append(dict(partition=part, referent=category, held=held,
                                         clip=f'{part}-{clip}', frame_idx=i))
    selected = sample_rows(pd.DataFrame(rows))
    assert selected.groupby(['partition', 'referent', 'held']).size().tolist() == [8] * 16 + [32] * 8
    assert selected.groupby(['partition', 'referent', 'clip']).size().max() <= 4
    assert selected.equals(sample_rows(pd.DataFrame(rows)))


def test_insufficient_distinct_clips_fails():
    with pytest.raises(ValueError, match='Insufficient'):
        sample_rows(pd.DataFrame([dict(partition='train', referent='bell', held=True, clip='x')]))


def test_panel_rejects_duplicate_situation_and_foil_clip_excess():
    from nursery.learner.cross_situational import validate_panels
    rows = pd.DataFrame([
        dict(partition='dev', referent='bell', clip='a', video_id='v', frame_idx=1),
        dict(partition='dev', referent='tube', clip='b', video_id='v', frame_idx=1),
        dict(partition='dev', referent='tube', clip='c', video_id='v', frame_idx=2),
        dict(partition='dev', referent='tube', clip='d', video_id='v', frame_idx=3),
    ])
    with pytest.raises(ValueError, match='repeats a physical situation'):
        validate_panels(rows, {'dev': [{'candidates': [0, 1, 2, 3]}], 'test': []})
    repeated = pd.concat([rows.iloc[[0]]] * 5, ignore_index=True)
    with pytest.raises(ValueError, match='More than four'):
        validate_panels(repeated, {'dev': [], 'test': []})


def test_prepare_refuses_overwrite_before_reading_inputs(tmp_path):
    from nursery.learner.cross_situational import prepare
    (tmp_path / 'panels.json').write_text('frozen')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        prepare('/missing/episodes', '/missing/cache', tmp_path)
    assert (tmp_path / 'panels.json').read_text() == 'frozen'
