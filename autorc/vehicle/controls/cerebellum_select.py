from autorc.vehicle.controls.cerebellum_advanced import CerebellumAdvanced
from autorc.vehicle.controls.cerebellum_basic import CerebellumBasic

def CerebellumSelect(type, update_interval_ms, controller, cortex, corti, model_name):

    if type == "ADVANCED":
        return CerebellumAdvanced(update_interval_ms, controller, cortex, corti, model_name)
    elif type == "BASIC":
        return CerebellumBasic(update_interval_ms, controller, cortex, corti)