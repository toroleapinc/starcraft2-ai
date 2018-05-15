"""Observation processing."""
import numpy as np

def preprocess_screen(screen, layers=None):
    if layers is None:
        layers = [5, 6, 7, 14, 15]
    out = screen[layers]
    for i in range(len(out)):
        m = out[i].max()
        if m > 0: out[i] /= m
    return out

def preprocess_minimap(minimap, layers=None):
    if layers is None:
        layers = [1, 4, 5]
    out = minimap[layers]
    for i in range(len(out)):
        m = out[i].max()
        if m > 0: out[i] /= m
    return out
