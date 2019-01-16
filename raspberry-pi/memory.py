import numpy as np

class Memory:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add(self, x, y):
        self.inputs.append([x])
        self.outputs.append([y])

    def save(self, name):
        np.save("data/"+name, np.array([self.inputs, self.outputs]))
        self.inputs = []
        self.outputs = []
