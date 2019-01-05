import numpy as np

class Memory:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add(self, x, y):
        self.inputs.append([x])
        self.outputs.append([y])

    def save(self):
        np.save("data", np.array([self.inputs, self.outputs]))
