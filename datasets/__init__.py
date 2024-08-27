from .scannetpp import build as build_scannetpp

def build_dataset(image_set, args):
    return build_scannetpp(image_set, args)
