import warnings
warnings.filterwarnings("ignore",category=UserWarning)

from utils import *
from dataset import DATASET_MAPPING
from interface import INTERFACE_MAPPING
from model import get_backbone, get_classifier, Net

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from pyheaven.torch_utils import HeavenDataset

from sound2synth import Sound2SynthModel, Identifier

if __name__=="__main__":
    args = HeavenArguments.from_parser([
        LiteralArgumentDescriptor("synth", short="s", choices=INTERFACE_MAPPING.keys(), default="Dexed"),
        LiteralArgumentDescriptor("dataset_type", short="dt", choices=DATASET_MAPPING.keys(), default="multimodal"),
        StrArgumentDescriptor("checkpoint", short="ckpt", default=None),
        StrArgumentDescriptor("dataset", short="ds", default="Dexed"),
        StrArgumentDescriptor("project", short="pj", default="Sound2Synth"),
        ListArgumentDescriptor("test_splits", short="test", type=str),

        StrArgumentDescriptor("backbone", short="bb", default="multimodal"),
        SwitchArgumentDescriptor("multimodal_use_spec", short="use_spec"),
        StrArgumentDescriptor("classifier", short="cl", default="parameter"),
        IntArgumentDescriptor("feature_dim", short="f", default=2048),
        IntArgumentDescriptor("batch_size", short="b", default=2),

        StrArgumentDescriptor("identifier", short="id", default=None),
        StrArgumentDescriptor("cuda", short="cd", default="0"),
        IntArgumentDescriptor("seed", short="sd", default=20001101),
        IntArgumentDescriptor("debug", default=-1),
        SwitchArgumentDescriptor("clean"),
        SwitchArgumentDescriptor("wandb",short='wandb'),
    ])
    if args.clean:
        CMD("rm -rf lightning_logs/*")
        CMD("rm -rf examples/*")
        CMD("rm -rf logs/*")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.n_gpu = len([d for d in args.cuda.split(',') if d.strip()!=''])

    seed_everything(args.seed, workers=True)
    
    # Prepare interface
    interface = INTERFACE_MAPPING[args.synth]

    # Prepare dataset
    args.datasets = MemberDict({
        'test' : DATASET_MAPPING[args.dataset_type](args.dataset,args.test_splits,with_params=True,augment=0.0),
    })
    if args.debug>=0:
        args.datasets['test'] = HeavenDataset(args.datasets['test'])[:args.debug]
    
    # Prepare network
    net = Net(
        backbone = get_backbone(args.backbone, args),
        classifier = get_classifier(args.classifier, interface, args),
    )

    # Prepare W&B logger
    args.identifier = Identifier(args)
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.project,
            entity='<WANDB_ENTITY>',
            log_model="all",
            id=args.identifier+"_test",
        )

    # Train model
    model = Sound2SynthModel(net, interface, args=args)
    assert(args.checkpoint is not None)
    if ExistFile(args.checkpoint):
        print(f"Loading checkpoint '{args.checkpoint}'.")
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    elif ExistFolder(args.checkpoint):
        ckpts = ListFiles(args.checkpoint,with_path=True,ordered=True)
        if len(ckpts)>0:
            print(f"Loading checkpoint '{ckpts[-1]}'.")
            model.load_state_dict(torch.load(ckpts[-1])['state_dict'])
        else:
            raise FileNotFoundError(args.checkpoint)
    else:
        raise FileNotFoundError(args.checkpoint)
    
    trainer = pl.Trainer(
        # GPU configuration
        gpus = 1, 
        auto_select_gpus = True,

        # Logging configuration
        logger = wandb_logger if args.wandb else True,
        log_every_n_steps = 100,
        
        # Speedup configuration
        benchmark = True,
    )

    if args.wandb:
        wandb_logger.watch(model)
    trainer.test(model)