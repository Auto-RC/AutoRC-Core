import numpy as np

class Memory:
    def __init__(self):
        self.mem = np.array([])

    def add(self, x, y):
        np.append(self.mem, [x, y])

    def save(self):
        np.save("data", self.mem)
        print(self.mem.size)
