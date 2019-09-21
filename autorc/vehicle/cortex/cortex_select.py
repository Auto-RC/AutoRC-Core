
from autorc.vehicle.cortex.cortex_advanced import CortexAdvanced
from autorc.vehicle.cortex.cortex_basic import CortexBasic

def CortexSelect(type, update_interval_ms, oculus, corti, controller):

        if type == "ADVANCED":
            return CortexAdvanced(update_interval_ms, oculus, corti, controller)
        elif type == "BASIC":
            return CortexBasic(update_interval_ms, oculus, corti, controller)
