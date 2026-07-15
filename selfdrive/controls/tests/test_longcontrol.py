from cereal import car, custom
from openpilot.selfdrive.controls.lib.longcontrol import STOP_HOLD_RELEASE_FRAMES, LongCtrlState, LongitudinalStopHold, long_control_state_trans




class TestLongControlStateTransition:

  def test_stay_stopped(self):
    CP = car.CarParams.new_message()
    CP_SP = custom.CarParamsSP.new_message()
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off


def test_physical_stop_rejects_false_launch_without_predicted_motion():
  stop_hold = LongitudinalStopHold()
  assert stop_hold.update(raw_should_stop=True, standstill=True, launch_intent=False)

  # Captured failure: shouldStop was false for 0.45 s, but the model still
  # predicted essentially zero motion. The held stop must not be released.
  for _ in range(10):
    assert stop_hold.update(raw_should_stop=False, standstill=True, launch_intent=False)


def test_physical_stop_requires_sustained_launch_intent():
  stop_hold = LongitudinalStopHold()
  assert stop_hold.update(raw_should_stop=True, standstill=True, launch_intent=False)

  for _ in range(STOP_HOLD_RELEASE_FRAMES - 1):
    assert stop_hold.update(raw_should_stop=False, standstill=True, launch_intent=True)
  assert not stop_hold.update(raw_should_stop=False, standstill=True, launch_intent=True)


def test_released_stop_does_not_reapply_without_a_new_stop_request():
  stop_hold = LongitudinalStopHold()
  for _ in range(STOP_HOLD_RELEASE_FRAMES):
    should_stop = stop_hold.update(raw_should_stop=False, standstill=True, launch_intent=True)
  assert not should_stop

  # The trajectory may dip below the launch threshold while the truck begins
  # moving. Only a new raw stop request may reapply the held stop.
  assert not stop_hold.update(raw_should_stop=False, standstill=True, launch_intent=False)
  assert stop_hold.update(raw_should_stop=True, standstill=True, launch_intent=False)

def test_engage():
  CP = car.CarParams.new_message()
  CP_SP = custom.CarParamsSP.new_message()
  active = True
  current_state = LongCtrlState.off
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid

def test_starting():
  CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
  CP_SP = custom.CarParamsSP.new_message()
  active = True
  current_state = LongCtrlState.starting
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.starting
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid
