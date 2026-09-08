from inference.temporal_state import TemporalButtonStates


def row(state, box=None):
    return dict(state=state,label='2',bbox=box or [0,0,100,100])


def test_on_immediate_and_off_requires_continuous_evidence():
    f=TemporalButtonStates(off_delay=.3)
    assert f.update([row('OFF')],0)[0]['temporal_state']=='OFF'
    assert f.update([row('ON')],.1)[0]['temporal_state']=='ON'
    for time in [.2,.3,.4]:
        assert f.update([row('OFF')],time)[0]['temporal_state']=='ON'
    assert f.update([row('OFF')],.51)[0]['temporal_state']=='OFF'


def test_different_buttons_never_share_state():
    f=TemporalButtonStates()
    result=f.update([row('ON'),row('OFF',[200,0,300,100])],0)
    assert [r['temporal_state'] for r in result]==['ON','OFF']
    result=f.update([row('OFF',[200,0,300,100]),row('OFF')],.1)
    assert [r['temporal_state'] for r in result]==['OFF','ON']
    assert result[0]['track_id']!=result[1]['track_id']


def test_lost_track_expires_and_missing_frame_does_not_create_box():
    f=TemporalButtonStates(max_gap=.2)
    original=f.update([row('ON')],0)[0]
    assert f.update([],.1)==[]
    new=f.update([row('OFF')],.3)[0]
    assert new['track_id']!=original['track_id']
    assert new['temporal_state']=='OFF'


def test_repeated_on_survives_long_false_off_but_eventually_turns_off():
    f=TemporalButtonStates(sustained_off_delay=1.2)
    f.update([row('ON')],0)
    assert f.update([row('ON')],.2)[0]['sustained_on']
    for k in range(3,15):
        result=f.update([row('OFF')],k/10)[0]
        assert result['temporal_state']=='ON'
    result=f.update([row('OFF')],1.51)[0]
    assert result['temporal_state']=='OFF'
    assert not result['sustained_on']


def test_isolated_on_does_not_get_extended_hold():
    f=TemporalButtonStates()
    f.update([row('ON')],0)
    for k in range(1,5):f.update([row('OFF')],k/10)
    assert f.update([row('OFF')],.5)[0]['temporal_state']=='OFF'


def test_unknown_does_not_keep_confirmed_on_for_long():
    f=TemporalButtonStates()
    f.update([row('ON')],0);f.update([row('ON')],.1)
    for k in range(2,6):f.update([row('INVALID')],k/10)
    assert f.update([row('INVALID')],.6)[0]['temporal_state']=='UNCERTAIN'
