import json
from pathlib import Path

import pytest

from nursery.simulator.closeup import validate_geometry, export


def test_geometry_preserves_distinct_same_category_instances():
    original = json.dumps([{'category':'ring','visible_px':2,'cells':[1]},
                           {'category':'ring','visible_px':1,'cells':[2]}])
    exported = [{'id':1,'visible_px':2,'image_xy':[2.5,3.5]},
                {'id':2,'visible_px':1,'image_xy':[5.5,6.5]}]
    result=json.loads(validate_geometry(original,exported))
    assert [r['id'] for r in result]==[1,2]
    assert [r['category'] for r in result]==['ring','ring']
    assert result[0]['image_xy'] != result[1]['image_xy']


@pytest.mark.parametrize('exported',[
    [{'id':1,'visible_px':2,'image_xy':[2.5,3.5]}],
    [{'id':2,'visible_px':1,'image_xy':[2.5,3.5]}],
    [{'id':1,'visible_px':1,'image_xy':None}],
    [],
])
def test_geometry_rejects_replay_drift(exported):
    with pytest.raises(ValueError):
        validate_geometry(json.dumps([{'category':'ring','visible_px':1}]),exported)


def test_invisible_object_has_no_invented_point():
    original=json.dumps([{'category':'ring','visible_px':0}])
    with pytest.raises(ValueError):
        validate_geometry(original,[{'id':1,'visible_px':0,'image_xy':[2.5,3.5]}])
    result=json.loads(validate_geometry(original,[{'id':1,'visible_px':0,'image_xy':None}]))
    assert result[0]['image_xy'] is None


def test_output_conflict_rejected_before_input_or_render(tmp_path, monkeypatch):
    import nursery.simulator.closeup as implementation
    monkeypatch.setattr(implementation,'_ignored_output',lambda path:Path(path))
    (tmp_path/'geometry.parquet').write_bytes(b'existing')
    monkeypatch.setattr(implementation,'_work',lambda *args:pytest.fail('Must reject before rendering'))
    with pytest.raises(FileExistsError):
        export(tmp_path/'missing_source',tmp_path,'geometry')
    assert (tmp_path/'geometry.parquet').read_bytes()==b'existing'


@pytest.mark.parametrize('duplicate',[False,True])
def test_gaze_keys_must_be_complete_and_unique(tmp_path,monkeypatch,duplicate):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import nursery.simulator.closeup as implementation
    source=tmp_path/'source';source.mkdir()
    pq.write_table(pa.Table.from_pylist([{'video_id':'s1_0000'}]),source/'sessions.parquet')
    pq.write_table(pa.Table.from_pylist([{'video_id':'s1_0000','frame_idx':0},
                                       {'video_id':'s1_0000','frame_idx':1}]),source/'frames.parquet')
    rows=[{'video_id':'s1_0000','frame_idx':0,'gaze_object_id':1,'gaze_x':.5,'gaze_y':.5}]
    if duplicate:rows*=2
    gaze=tmp_path/'gaze.parquet';pq.write_table(pa.Table.from_pylist(rows),gaze)
    monkeypatch.setattr(implementation,'_ignored_output',lambda path:Path(path))
    monkeypatch.setattr(implementation,'_work',lambda *args:pytest.fail('Invalid gaze must not render'))
    with pytest.raises(ValueError,match='Duplicate' if duplicate else 'cover'):
        export(source,tmp_path/'output','closeup',gaze_path=gaze)
