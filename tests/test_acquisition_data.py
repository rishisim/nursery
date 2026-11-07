import json

import numpy as np
import pandas as pd
import pytest

from nursery.learner.acquisition_data import _panel, _source_keys


def test_source_identity_rejects_shared_name_and_duplicate_frames():
    frames = pd.DataFrame({'video_id': ['a', 'b'], 'frame_idx': [0, 0],
                           'clip': ['same', 'same'], 'subject': ['s1', 's2']})
    with pytest.raises(ValueError, match='shared across subjects'):
        _source_keys(frames)
    frames.loc[1, 'subject'] = 's1'
    frames.loc[1, 'video_id'] = 'a'
    with pytest.raises(ValueError, match='Duplicate'):
        _source_keys(frames)


def test_panel_uses_noun_query_and_different_source_donors():
    categories = ['ball', 'cup', 'dog', 'cat']
    frames, rows = [], []
    for i in range(8):
        category = categories[i % 4]
        frames.append({'video_id': f'v{i}', 'frame_idx': 0, 'clip': f's1/c{i}', 'subject': 's1',
                       'objects': json.dumps([{'category': category, 'visible_px': 1}]), 'held_category': None})
        rows.append({'video_id': f'v{i}', 'frame_idx': 0, 'referent': category,
                     'text': f'a {category}', 'named': True, 'referent_visible': True})
    frame = pd.DataFrame(frames)
    lut = {f'v{i}|0': (0, i) for i in range(8)}
    panel, trials = _panel(pd.DataFrame(rows), frame, lut)
    assert panel.text.tolist() == panel.referent.tolist()
    assert len(trials['trial_rows']) == 8
    for i, donor in enumerate(trials['donor_rows']):
        assert frame.iloc[i].clip != frame.iloc[donor].clip
    for row, candidates, correct in zip(trials['trial_rows'], trials['candidate_rows'], trials['correct_slots']):
        assert candidates[correct] == row
        assert len(set(candidates)) == 4
    assert _panel(pd.DataFrame(rows), frame, lut)[1] == trials


def test_picture_panel_is_fixed_before_vocabulary():
    from nursery.learner.acquisition_data import _picture_panel

    rows = pd.DataFrame([{'video_id': f'{c}{i}', 'frame_idx': 0, 'category': c}
                         for c in ['ball', 'cup', 'cat', 'dog'] for i in range(4)])
    index = rows[['video_id', 'frame_idx']].copy()
    index['row'] = np.arange(len(index))
    panel = _picture_panel(rows, index)
    assert len(panel['trials']) == 400
    assert panel == _picture_panel(rows, index)
    for trial in panel['trials']:
        row = trial['candidate_rows'][trial['correct_slot']]
        assert rows.iloc[row]['category'] == trial['query']
        assert len({rows.iloc[i]['category'] for i in trial['candidate_rows']}) == 4


def test_audit_rejects_wrong_unknown_query_credit(tmp_path):
    from nursery.learner.acquisition_data import audit

    (tmp_path / 'protocol.json').write_text(json.dumps({'unknown_query_score': 0.0}))
    with pytest.raises(ValueError, match='Unknown query credit'):
        audit(tmp_path)
