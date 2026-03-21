from preview_navigation import clamp_frame_index, frame_from_slider_click, step_frame_index


def test_frame_from_slider_click_maps_position_and_clamps_ratio():
    assert frame_from_slider_click(x=50, width=100, from_value=0, to_value=20) == 10
    assert frame_from_slider_click(x=-10, width=100, from_value=0, to_value=20) == 0
    assert frame_from_slider_click(x=200, width=100, from_value=0, to_value=20) == 20


def test_clamp_frame_index_limits_to_valid_range():
    assert clamp_frame_index(-3, 12) == 0
    assert clamp_frame_index(7, 12) == 7
    assert clamp_frame_index(99, 12) == 12


def test_step_frame_index_applies_delta_then_clamps():
    assert step_frame_index(current=5, delta=2, max_frame=12) == 7
    assert step_frame_index(current=0, delta=-1, max_frame=12) == 0
    assert step_frame_index(current=11, delta=5, max_frame=12) == 12
