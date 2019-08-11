from autorc.vehicle.controls.cerebellum_advanced import CerebellumAdvanced
from autorc.vehicle.controls.cerebellum_basic import CerebellumBasic

class CerebellumSelect():

    @staticmethod
    def select(type, update_interval_ms, controller, cortex, corti):

        if type == "ADVANCED":
            return CerebellumAdvanced(update_interval_ms, controller, cortex, corti)
        elif type == "BASIC":
            return CerebellumBasic(update_interval_ms, controller, cortex, corti)