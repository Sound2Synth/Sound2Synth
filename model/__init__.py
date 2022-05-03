from .main_backbone import MainBackbone
from .linear_backbone import LinearBackbone
from .conv_backbone import ConvBackbone
from .lstm_backbone import LSTMBackbone
from .convmixer_backbone import ConvMixerBackbone
from .multimodal_backbone import MultimodalBackbone
from .pdc_backbone import VCNN2Backbone, PDCNN2Backbone
from pyheaven.torch_utils import TimmBackbone
from .net import NaiveParametersClassifier, OscillatorAttentionClassifier, Net

def get_backbone(backbone_type, args):
    feature_dim = args.feature_dim
    if "linear" in backbone_type:
        backbone = LinearBackbone(hidden_dim=feature_dim,output_dim=feature_dim)
    elif "convmixer" in backbone_type:
        backbone = ConvMixerBackbone(output_dim=feature_dim)
    elif "conv" in backbone_type:
        backbone = ConvBackbone(output_dim=feature_dim)
    elif "lstm" in backbone_type:
        backbone = LSTMBackbone(output_dim=feature_dim)
    elif "res" in backbone_type:
        backbone = MainBackbone(TimmBackbone(model_type=backbone_type,
                                input_layer_getter=lambda n:n.conv1,modify_input_channels=1,
                                output_layer_name="fc",embedding_dim=feature_dim,pretrained=True))
    elif "dense" in backbone_type:
        backbone = MainBackbone(TimmBackbone(model_type=backbone_type,
                                input_layer_getter=lambda n:n.features.conv0,modify_input_channels=1,
                                output_layer_name="classifier",embedding_dim=feature_dim,pretrained=True))
    elif "multimodal" in backbone_type:
        backbone = MultimodalBackbone(output_dim=feature_dim,mode_dropout=args.mode_dropout,conv=args.multimodal_use_spec)
    elif "pdcnn" in backbone_type:
        backbone = PDCNN2Backbone(output_dim=feature_dim)
    elif "vcnn" in backbone_type:
        backbone = VCNN2Backbone(output_dim=feature_dim)
    else:
        raise NotImplementedError
    return backbone

def get_classifier(classifier_type, interface, args):
    feature_dim = args.feature_dim
    if classifier_type=='parameter':
        classifier = NaiveParametersClassifier(
            division = interface.division(),
            input_dim = feature_dim,
        )
    elif classifier_type=='oscillator':
        oscillator_names = ['OP1','OP2','OP3','OP4','OP5','OP6']; oscillators = [[],[],[],[],[],[],[]]
        named_division = interface.named_division(); book = set()
        for osc, osc_name in zip(oscillators, oscillator_names):
            for param_id, param in enumerate(named_division):
                if osc_name in param:
                    osc.append(param_id); book.add(param_id)
        for i in range(len(named_division)):
            if i not in book:
                oscillators[-1].append(i)
        classifier = OscillatorAttentionClassifier(
            named_division = named_division,
            oscillators = oscillators,
            input_dim = feature_dim,
        )
    else:
        raise NotImplementedError
    return classifier