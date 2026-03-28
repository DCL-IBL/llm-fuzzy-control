import torch

def triangular(x, a, b, c):
    # a < b < c (can be tensors or scalars, broadcastable with x)
    left  = (x - a) / (b - a)
    right = (c - x) / (c - b)
    return torch.clamp(torch.minimum(left, right), min=0.0, max=1.0)

def ramp_right(x, a, b):
    y = (x - a) / (b - a)
    return torch.clamp(y, min=0.0, max=1.0)

def ramp_left(x, a, b):
    y = - (x - b) / (b - a)
    return torch.clamp(y, min=0.0, max=1.0)

def hypo_glucose(x):
    return ramp_left(x,90/18,105/18) # 70 - 105 mg/dl

def target_glucose(x):
    return triangular(x,90/18,105/18,180/18) # 70 - 180 mg/dl

def hyper_glucose(x):
    return ramp_right(x, 150/18, 300/18) # 105 - 180 mg/dl

def zero_dose(x):
    return ramp_left(x,0,0.01) # mU/kg

def low_dose(x):
    return triangular(x, 0.0, 0.01, 0.1) # mU/kg

def high_dose(x):
    return ramp_right(x, 0.05, 0.2) # mU/kg

def bolus_dose(x):
    return ramp_right(x, 0.1, 1.5) # mU/kg