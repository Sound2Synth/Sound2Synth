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
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint, EarlyStopping

from sound2synth import Sound2SynthModel, SplitDatasets, Identifier

if __name__=="__main__":
    args = HeavenArguments.from_parser([
        LiteralArgumentDescriptor("synth", short="s", choices=INTERFACE_MAPPING.keys(), default="Dexed"),
        LiteralArgumentDescriptor("dataset_type", short="dt", choices=DATASET_MAPPING.keys(), default="multimodal"),
        StrArgumentDescriptor("dataset", short="ds", default="Dexed"),
        StrArgumentDescriptor("project", short="pj", default="Sound2Synth"),
        ListArgumentDescriptor("train_splits", short="train", type=str),
        ListArgumentDescriptor("val_splits", short="val", type=str),
        ListArgumentDescriptor("test_splits", short="test", type=str),
        SwitchArgumentDescriptor("split_train_to_val", short="split2val"),
        SwitchArgumentDescriptor("split_train_to_test", short="split2test"),

        StrArgumentDescriptor("backbone", short="bb", default="multimodal"),
        SwitchArgumentDescriptor("multimodal_use_spec", short="use_spec"),
        StrArgumentDescriptor("classifier", short="cl", default="parameter"),
        IntArgumentDescriptor("feature_dim", short="f", default=2048),
        IntArgumentDescriptor("num_epochs", short="e", default=30),
        IntArgumentDescriptor("batch_size", short="b", default=2),
        IntArgumentDescriptor("grad_accum", short="ga", default=-64),
        IntArgumentDescriptor("examples", short="ex", default=128),

        FloatArgumentDescriptor("limit_train_batches", short="limtr", default=1.0),
        FloatArgumentDescriptor("limit_val_batches", short="limvl", default=1.0),
        FloatArgumentDescriptor("learning_rate", short="lr", default=2e-4),
        FloatArgumentDescriptor("warmup_start_lr_ratio", short="wlr", default=.01),
        FloatArgumentDescriptor("eta_min", short="em", default=1e-8),
        IntArgumentDescriptor("warmup_epochs", short="we", default=4),
        FloatArgumentDescriptor("weight_decay", short="wd", default=1e-4),
        FloatArgumentDescriptor("noise_augment", short="na", default=1e-4),
        FloatArgumentDescriptor("mode_dropout", short="dp", default=0.0),

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
    if args.grad_accum < 0:
        args.grad_accum = -args.grad_accum//args.batch_size
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.n_gpu = len([d for d in args.cuda.split(',') if d.strip()!=''])

    seed_everything(args.seed, workers=True)
    
    # Prepare interface
    interface = INTERFACE_MAPPING[args.synth]
    
    # Prepare network
    net = Net(
        backbone = get_backbone(args.backbone, args),
        classifier = get_classifier(args.classifier, interface, args),
    )
    
    # Save Arguments
    args.identifier = Identifier(args)
    SaveJson(args,pjoin("tasks", f"{args.identifier}.json"),indent=4)

    # Prepare dataset
    dataset = DATASET_MAPPING[args.dataset_type](args.dataset,args.train_splits,with_params=True,augment=args.noise_augment)
    if args.split_train_to_val and args.split_train_to_test:
        args.datasets = SplitDatasets(dataset,train=0.7,val=0.2,test=0.1,cut=args.debug)
    elif args.split_train_to_val:
        args.datasets = SplitDatasets(dataset,train=0.8,val=0.2,test=0.0,cut=args.debug)
        args.datasets['test'] = DATASET_MAPPING[args.dataset_type](args.dataset,args.test_splits,with_params=True,augment=args.noise_augment)
    elif args.split_train_to_test:
        args.datasets = SplitDatasets(dataset,train=0.9,val=0.0,test=0.1,cut=args.debug)
        args.datasets['val'] = DATASET_MAPPING[args.dataset_type](args.dataset,args.val_splits,with_params=True,augment=args.noise_augment)
    else:
        args.datasets = MemberDict({
            'train': dataset,
            'val'  : DATASET_MAPPING[args.dataset_type](args.dataset,args.val_splits,with_params=True,augment=args.noise_augment),
            'test' : DATASET_MAPPING[args.dataset_type](args.dataset,args.test_splits,with_params=True,augment=args.noise_augment),
        })

    # Prepare W&B logger
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.project,
            entity='<WANDB_ENTITY>',
            log_model="all",
            id=args.identifier,
        )

    # Train model
    model = Sound2SynthModel(net, interface, args=args)
    trainer = pl.Trainer(
        max_epochs = args.num_epochs,
        gradient_clip_val = 1.0,
        accumulate_grad_batches = args.grad_accum,
        callbacks = [
            StochasticWeightAveraging(),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(monitor='valid_celoss',mode='min'),
            EarlyStopping(monitor='valid_celoss',mode='min',patience=8,check_finite=True),
        ],

        # GPU configuration
        gpus = args.n_gpu, 
        auto_select_gpus = True,

        # Logging configuration
        logger = wandb_logger if args.wandb else True,
        log_every_n_steps = 100,
        
        # Speedup configuration
        benchmark = True,
        strategy = DDPPlugin(find_unused_parameters=(args.model=='group')),
        limit_train_batches = args.limit_train_batches,
        limit_val_batches = args.limit_val_batches,

        # Tuning configuration
        auto_lr_find = False,   # 2e-4
    )

    if args.wandb:
        wandb_logger.watch(model)
    trainer.fit(model)
    trainer.test(ckpt_path="best")