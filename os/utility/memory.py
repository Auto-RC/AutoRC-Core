import numpy as np

class Memory:

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add(self, x, y):
        if y == [0, 0]:
            print("skipped")
        else:
            self.inputs.append([x])
            self.outputs.append([y])
            print("len x:", len(self.inputs))
            print("len y:", len(self.outputs))

    def save(self, name):
        np.save("data/"+name, np.array([self.inputs, self.outputs]))
        self.inputs = []
        self.outputs = []
