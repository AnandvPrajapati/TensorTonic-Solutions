import math

def log_transform(values):
    arr = []
    for v in values:
        y = math.log(1 + v)
        arr.append(y)
    return arr