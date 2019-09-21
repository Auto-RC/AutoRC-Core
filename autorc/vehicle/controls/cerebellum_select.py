from autorc.vehicle.controls.cerebellum_reinforcement_learning import CerebellumSupervisedLearning
from autorc.vehicle.controls.cerebellum_standard_controls import CerebellumStandardControls

def CerebellumSelect(type, update_interval_ms, controller, cortex, corti, model_name):

    if type == "ADVANCED":
        return CerebellumSupervisedLearning(update_interval_ms, controller, cortex, corti, model_name)
    elif type == "BASIC":
        return CerebellumStandardControls(update_interval_ms, controller, cortex, corti)
