from autorc.vehicle.controls.cerebellum_reinforcement_learning import CerebellumSupervisedLearning
from autorc.vehicle.controls.cerebellum_standard_controls import CerebellumStandardControls

class CerebellumSelect():

    @staticmethod
    def select(type, update_interval_ms, controller, cortex, corti):

        if type == "ADVANCED":
            return CerebellumSupervisedLearning(update_interval_ms, controller, cortex, corti)
        elif type == "BASIC":
            return CerebellumStandardControls(update_interval_ms, controller, cortex, corti)