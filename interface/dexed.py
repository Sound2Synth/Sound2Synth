from .base import *

# 87 regression parameters
# 66 classification parameters
# 2 fixed parameters: 'ALGORITHM' and 'Output'
REG_NCLASS = 64
regression_parameters = [
    'Cutoff', 'LFO AM DEPTH', 'LFO DELAY', 'LFO PM DEPTH', 'LFO SPEED', 'MASTER TUNE ADJ',
    'OP1 EG LEVEL 1', 'OP1 EG LEVEL 2', 'OP1 EG LEVEL 3', 'OP1 EG LEVEL 4', 'OP1 EG RATE 1', 'OP1 EG RATE 2', 'OP1 EG RATE 3', 'OP1 EG RATE 4', 'OP1 F FINE', 'OP1 L SCALE DEPTH', 'OP1 OUTPUT LEVEL', 'OP1 R SCALE DEPTH',
    'OP2 EG LEVEL 1', 'OP2 EG LEVEL 2', 'OP2 EG LEVEL 3', 'OP2 EG LEVEL 4', 'OP2 EG RATE 1', 'OP2 EG RATE 2', 'OP2 EG RATE 3', 'OP2 EG RATE 4', 'OP2 F FINE', 'OP2 L SCALE DEPTH', 'OP2 OUTPUT LEVEL', 'OP2 R SCALE DEPTH',
    'OP3 EG LEVEL 1', 'OP3 EG LEVEL 2', 'OP3 EG LEVEL 3', 'OP3 EG LEVEL 4', 'OP3 EG RATE 1', 'OP3 EG RATE 2', 'OP3 EG RATE 3', 'OP3 EG RATE 4', 'OP3 F FINE', 'OP3 L SCALE DEPTH', 'OP3 OUTPUT LEVEL', 'OP3 R SCALE DEPTH',
    'OP4 EG LEVEL 1', 'OP4 EG LEVEL 2', 'OP4 EG LEVEL 3', 'OP4 EG LEVEL 4', 'OP4 EG RATE 1', 'OP4 EG RATE 2', 'OP4 EG RATE 3', 'OP4 EG RATE 4', 'OP4 F FINE', 'OP4 L SCALE DEPTH', 'OP4 OUTPUT LEVEL', 'OP4 R SCALE DEPTH',
    'OP5 EG LEVEL 1', 'OP5 EG LEVEL 2', 'OP5 EG LEVEL 3', 'OP5 EG LEVEL 4', 'OP5 EG RATE 1', 'OP5 EG RATE 2', 'OP5 EG RATE 3', 'OP5 EG RATE 4', 'OP5 F FINE', 'OP5 L SCALE DEPTH', 'OP5 OUTPUT LEVEL', 'OP5 R SCALE DEPTH',
    'OP6 EG LEVEL 1', 'OP6 EG LEVEL 2', 'OP6 EG LEVEL 3', 'OP6 EG LEVEL 4', 'OP6 EG RATE 1', 'OP6 EG RATE 2', 'OP6 EG RATE 3', 'OP6 EG RATE 4', 'OP6 F FINE', 'OP6 L SCALE DEPTH', 'OP6 OUTPUT LEVEL', 'OP6 R SCALE DEPTH',
    'PITCH EG LEVEL 1', 'PITCH EG LEVEL 2', 'PITCH EG LEVEL 3', 'PITCH EG LEVEL 4', 'PITCH EG RATE 1', 'PITCH EG RATE 2', 'PITCH EG RATE 3', 'PITCH EG RATE 4', 'Resonance'
]
classification_parameters = [
    ('FEEDBACK', 8), ('LFO KEY SYNC', 2), ('LFO WAVE', 6), ('OSC KEY SYNC', 2),
    ('OP1 A MOD SENS.', 4), ('OP1 F COARSE', 32), ('OP1 KEY VELOCITY', 8), ('OP1 L KEY SCALE', 4), ('OP1 MODE', 2), ('OP1 OSC DETUNE', 15), ('OP1 R KEY SCALE', 4), ('OP1 RATE SCALING', 8), ('OP1 BREAK POINT', 88), ('OP1 SWITCH', 2),
    ('OP2 A MOD SENS.', 4), ('OP2 F COARSE', 32), ('OP2 KEY VELOCITY', 8), ('OP2 L KEY SCALE', 4), ('OP2 MODE', 2), ('OP2 OSC DETUNE', 15), ('OP2 R KEY SCALE', 4), ('OP2 RATE SCALING', 8), ('OP2 BREAK POINT', 88), ('OP2 SWITCH', 2),
    ('OP3 A MOD SENS.', 4), ('OP3 F COARSE', 32), ('OP3 KEY VELOCITY', 8), ('OP3 L KEY SCALE', 4), ('OP3 MODE', 2), ('OP3 OSC DETUNE', 15), ('OP3 R KEY SCALE', 4), ('OP3 RATE SCALING', 8), ('OP3 BREAK POINT', 88), ('OP3 SWITCH', 2),
    ('OP4 A MOD SENS.', 4), ('OP4 F COARSE', 32), ('OP4 KEY VELOCITY', 8), ('OP4 L KEY SCALE', 4), ('OP4 MODE', 2), ('OP4 OSC DETUNE', 15), ('OP4 R KEY SCALE', 4), ('OP4 RATE SCALING', 8), ('OP4 BREAK POINT', 88), ('OP4 SWITCH', 2),
    ('OP5 A MOD SENS.', 4), ('OP5 F COARSE', 32), ('OP5 KEY VELOCITY', 8), ('OP5 L KEY SCALE', 4), ('OP5 MODE', 2), ('OP5 OSC DETUNE', 15), ('OP5 R KEY SCALE', 4), ('OP5 RATE SCALING', 8), ('OP5 BREAK POINT', 88), ('OP5 SWITCH', 2),
    ('OP6 A MOD SENS.', 4), ('OP6 F COARSE', 32), ('OP6 KEY VELOCITY', 8), ('OP6 L KEY SCALE', 4), ('OP6 MODE', 2), ('OP6 OSC DETUNE', 15), ('OP6 R KEY SCALE', 4), ('OP6 RATE SCALING', 8), ('OP6 BREAK POINT', 88), ('OP6 SWITCH', 2),
    ('P MODE SENS.', 8), ('TRANSPOSE', 49)
]
def get_dexed_descriptors():
    descriptors = MemberDict()
    descriptors |= {
        key: MemberDict({
            "serialize": lambda x:(x['value']-x['param'].min_value)/(x['param'].max_value-x['param'].min_value),
            "unserialize": lambda x:x['value']*(x['param'].max_value-x['param'].min_value)+x['param'].min_value,
        }) for key in regression_parameters
    }
    descriptors |= {
        key: MemberDict({
            "serialize": lambda x:x['value'],
            "unserialize": lambda x:x['value'],
        }) for key, classes in classification_parameters
    }
    descriptors |= {
        "ALGORITHM": MemberDict({
            "serialize": lambda x:None,
            "unserialize": lambda x:x['param'].current_value if x['interface'].algorithm is None else x['interface'].algorithm,
        })
    }
    descriptors |= {
        "Output": MemberDict({
            "serialize": lambda x:None,
            "unserialize": lambda x:1.0,
        })
    }
    return descriptors

class DexedInterface(BaseInterface):
    regression_nclasses = REG_NCLASS
    regression_parameters = regression_parameters
    classification_parameters = classification_parameters
    ordered_descriptors = get_dexed_descriptors()
    criteria = ParameterSpaceLoss(
        regression_parameters=regression_parameters,
        classification_parameters=classification_parameters,
        regression_nclasses=REG_NCLASS,
    )
    
    def __init__(self, *args, **kwargs):
        super(DexedInterface, self).__init__(*args, **kwargs); self.set_algorithm(None)

    def set_algorithm(self, algorithm=None):
        self.algorithm = algorithm