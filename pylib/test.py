import diffcloth_py as df
print(df.__file__)
import sys

config = {
    'fabric': {
        "clothDimX" : 13,
        "clothDimY" : 13,
        "k_stiff_stretching" : 3000, 
        "k_stiff_bending" :  0.3,
        "gridNumX" : 40,
        "gridNumY" : 80,
        "density" : 0.3,
        "keepOriginalScalePoint" : False,
        'isModel' : True,
        "custominitPos" : False,
        "fabricIdx" : 1,  # Enum Value
        "color" : (0.9, 0.9, 0.9),
        "name" :  "remeshed/dress-handsup-drape.obj",
    },
    'scene' : {
        "orientation" : 0, # Enum Value
        "attachmentPoints" : 2, # CUSTOM_ARRAY
        "customAttachmentVertexIdx": [(0., [2,3,4])], 
        "trajectory" : 0, # Enum Value
        "primitiveConfig" : 0, # Enum Value
        'windConfig' : 2, # Enum Value
        'camPos' :  (-10.38, 4.243, 12.72),
        "camFocusPos" : (0, 0, 0),
        'camFocusPointType' : 3, # Enum Value
        "sceneBbox" :  {"min": (-7, -7, -7), "max": (7, 7, 7)},
        "timeStep" : 1.0 / 90.0,
        "stepNum" : 250,
        "forwardConvergenceThresh" : 1e-8,
        'backwardConvergenceThresh' : 5e-4,
        'name' : "wind_tshirt"
    }
}

if __name__ == '__main__':
    print(sys.path)
    df.makeSimFromConfig(config)
    pass