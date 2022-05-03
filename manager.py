from utils import *
from interface.dexed import DexedInterface
from pyheaven.plot_utils import *

_HEADER = """export PYTHONPATH="./"
export QT_MAC_WANTS_LAYER=1
"""
_TRAIN = """python3 train.py --synth Dexed --dataset Dexed --clean \\
                 --dataset-type {0} --backbone {1} --classifier {6} --batch-size {2} \\
                 --train-splits {3} --split-train-to-val --limit-train-batches {7} --limit-val-batches {8} \\
                 --test-splits {4} --identifier {5} $@
"""
_TEST = """python3  test.py --synth Dexed --dataset Dexed \\
                 --dataset-type {0} --backbone {1} --classifier {6} --batch-size {2} \\
                 --checkpoint ./experiment/{5}/checkpoints/ \\
                 --test-splits {4} --identifier {5} $@
"""
TASK_TEST_TEMPLATE = _HEADER + _TEST
TASK_TEMPLATE = _HEADER + _TRAIN + _TEST
BACKBONE_DATASETS = {
    'spec': ("spec","conv",4),
    'multi': ("multimodal","multimodal",2),
    'multi-spec': ("multimodal","multimodal --multimodal-use-spec",1),
}
SPLITS = {
    "large": "LARGE{0:02d}",
    "aug": "LARGE{0:02d} AUG{0:02d}",
    "rand": "LARGE{0:02d} RAND{0:02d}",
    "aug-rand": "LARGE{0:02d} AUG{0:02d} RAND{0:02d}",
}
RSYNC_INTERVAL = 1200

def RenderFolder(folder, S):
    for file in ListFiles(folder):
        path = pjoin(folder, file)
        if not ExistFile(AsFormat(path,"wav")):
            try:
                params = DexedInterface.from_dict(LoadJson(path)); params.to_synth(S)
                audio, sample_rate = S.render(); tau.save(AsFormat(path,"wav"),audio,sample_rate)
            except Exception as e:
                PrintError(f"{FORMATTED_TIME()}: Render error. Skipped. \"{file}\"")

def Monitor(server, identifier, inference, S):
    root = p2s(pjoin("experiment",identifier),f=True)
    RSYNC(pjoin(server,"Sound2Synth","examples"),root,args="-r -vP -q")
    valid_stat = LoadJson(pjoin(root,"valid.json")) if ExistFile(pjoin(root,"valid.json")) else {}
    for epoch in ListFolders(pjoin(root,"examples")):
        if (epoch not in valid_stat)\
        or ('count' not in valid_stat[epoch])\
        or len([f for f in ListFiles(pjoin(root,"examples",epoch)) if f.endswith("_gt.json")])>valid_stat[epoch]['count']\
        or inference:
            RenderFolder(pjoin(root,"examples",epoch), S)
            valid_stat[epoch] = Evaluate(pjoin(root,"examples",epoch),tqdm=True)
            SaveJson(valid_stat,pjoin(root,"valid.json"),indent=4)
    SaveJson(valid_stat,pjoin(root,"valid.json"),indent=4)
    if not inference:
        return
    try:
        RSYNC(p2s(pjoin(server,"Sound2Synth","inference",identifier),f=True)+"*",p2s(pjoin(root,"inference"),f=True),args="-r -vP -q")
        RenderFolder(pjoin(root,"inference"), S)
        inference_stat = Evaluate(pjoin(root,"inference"),tqdm=True)
        SaveJson(inference_stat,pjoin(root,"inference.json"),indent=4)
    except Exception as e:
        print("Inference not finished!")
        print(e)

def Summary(root, metrics=["AudioMFCCD"]):
    plt.rc('font', family='Open Sans')
    valid_stat = LoadJson(pjoin(root,"valid.json"))
    for metric in metrics:
        scores = [valid_stat["%05d"%epoch][metric]/valid_stat["%05d"%epoch]['count'] for epoch in range(len(valid_stat))]
        with Plotter(figsize=(16,9),path=pjoin(root,f"{metric}.pdf"),legend=False) as (fig,axe):
            x = list(range(len(scores))); y = scores
            line = plt.plot(
                x, y, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,
            )[0]
            points = [tuple(point) for point in line.get_xydata()]
            for point in points:
                axe.text(
                    point[0],
                    point[1] + 0.5,
                    round(point[1], 1),
                    horizontalalignment='center',
                    color='black',
                    weight='bold',
                    fontsize=18,
                )
            axe.spines['top'].set_visible(False)
            axe.spines['right'].set_visible(False)
            axe.spines['bottom'].set_visible(False)
            axe.spines['left'].set_visible(False)
            axe.tick_params(axis='both', which='major', labelsize=14)
            axe.set_xlabel("# Epochs", labelpad=15, color='#333333', weight='bold', fontsize=18)
            axe.set_ylabel(f"Average Validation {metric}", labelpad=15, color='#333333', weight='bold', fontsize=18)
            axe.set_title(f"{metric}-Epoch Curve", pad=15, color='#333333', weight='bold', fontsize=22)
            fig.tight_layout()

if __name__=="__main__":
    args = HeavenArguments.from_parser([
        IntArgumentDescriptor("algorithm",default=0),
        StrArgumentDescriptor("server",default=get_config('server_root')),
        StrArgumentDescriptor("data-server",short='ds',default=get_config('server_root')),
        LiteralArgumentDescriptor("split",short="sp",choices=['large','aug','rand','aug-rand'],default='large'),
        LiteralArgumentDescriptor("backbone",choices=['spec','multi','multi-spec'],default='spec'),
        LiteralArgumentDescriptor("classifier",choices=['parameter','oscillator'],default='parameter'),
        StrArgumentDescriptor("exp-info",short='info',default=None),
        FloatArgumentDescriptor("limit_train_batches", short="limtr", default=1.0),
        FloatArgumentDescriptor("limit_val_batches", short="limvl", default=1.0),
        SwitchArgumentDescriptor("sync-data",short='syncd'),
        SwitchArgumentDescriptor("monitor"),
        SwitchArgumentDescriptor("no-metric",short="nom"),
        SwitchArgumentDescriptor("add-to-ensemble",short="add"),
        SwitchArgumentDescriptor("back-sync",short="backs"),
        SwitchArgumentDescriptor("clear",short="clr"),
        StrArgumentDescriptor("conda-env",short="conda",default="source /home/<USERNAME>/miniconda3/bin/activate; conda activate base"),
        # StrArgumentDescriptor("conda-env",short="conda",default="conda activate py38"),
        StrArgumentDescriptor("cuda", short="cd", default="0"),
    ])
    print("Initializing ...")
    assert(args.algorithm!=None)
    CreateFolder("experiment"); CreateFolder("examples"); CreateFolder("inference")
    experiment = f"{args.backbone}_{args.split}_{args.algorithm}{'' if args.exp_info is None else '_'+args.exp_info}"
    root = pjoin("experiment",experiment)
    ClearFolder(root) if args.clear else CreateFolder(root)
    CreateFolder(pjoin(root,"examples")); CreateFolder(pjoin(root,"inference"))
    handle = None
    print(f"Experiment '{experiment}' initialized.")

    if not args.monitor:
        print("Syncing code to server ...")
        SaveJson(args,pjoin(root,"args.json"),indent=4)
        RSYNC("../Sound2Synth",args.server,args="-r -vP -q --exclude '*.wav'")
        server_code_dir = pjoin(args.server,"Sound2Synth")
        if args.sync_data:
            print("Syncing data to server ...")
            CMD(f"ssh {args.server.split(':')[0]} \"rsync -r -vP -q {pjoin(args.data_server,'data')} ./\"")
        server_data_dir = pjoin(args.server,'data')

        print("Creating scripts ...")
        task_script_dir = f"tasks/{experiment}.sh"
        with open(task_script_dir,"w") as f:
            B = BACKBONE_DATASETS[args.backbone]; S = SPLITS[args.split]
            f.write(TASK_TEMPLATE.format(
                B[0], B[1], B[2], S.format(args.algorithm), "TEST{:02d}".format(args.algorithm), experiment, args.classifier, args.limit_train_batches, args.limit_val_batches
            ))
        RSYNC(task_script_dir,pjoin(server_code_dir,task_script_dir),args="-r -vP -q")
        task_test_script_dir = f"tasks/{experiment}.test.sh"
        with open(task_test_script_dir,"w") as f:
            B = BACKBONE_DATASETS[args.backbone]; S = SPLITS[args.split]
            f.write(TASK_TEST_TEMPLATE.format(
                B[0], B[1], B[2], S.format(args.algorithm), "TEST{:02d}".format(args.algorithm), experiment, args.classifier, args.limit_train_batches, args.limit_val_batches
            ))
        RSYNC(task_test_script_dir,pjoin(server_code_dir,task_test_script_dir),args="-r -vP -q")
        print("Experiment running ...")
        run_command = f"ssh {args.server.split(':')[0]} \"cd ./Sound2Synth && {args.conda_env+' && ' if args.conda_env is not None else ' '}bash {task_script_dir} --wandb --cuda {args.cuda}\""
        handle = CMD(run_command,wait=False)

    if not args.no_metric:
        try:
            with Synthesizer("Dexed") as S:
                if not args.monitor:
                    while handle.poll() is None:
                        print("Monitoring ...")
                        Monitor(args.server, experiment, inference=False, S=S)
                        time.sleep(RSYNC_INTERVAL)
                Monitor(args.server, experiment, inference=True, S=S)
            Summary(root)
        except Exception as e:
            print(e)
            with Synthesizer("Dexed") as S:
                Monitor(args.server, experiment, inference=True, S=S)
            Summary(root)

    print("Downloading checkpoints ...")
    try:
        RSYNC(pjoin(args.server, "Sound2Synth", "Sound2Synth", experiment, "checkpoints"), p2s(root, f=True))
        checkpoints = pjoin(root, "checkpoints"); latest_checkpoint = pjoin(checkpoints, "latest")
        latest_name = ListFiles(checkpoints,ordered=True,key=BY_CTIME_CRITERIA)[-1]
        CopyFile(pjoin(checkpoints, latest_name), pjoin(checkpoints, "latest"))
    except Exception as e:
        print(e)

    print("Downloading experiment data ...")
    try:
        RSYNC(pjoin(args.server, "Sound2Synth", "tasks"), "./")
        args.train_args = LoadJson(pjoin("tasks", f"{experiment}.json"))
        SaveJson(args,pjoin(root,"args.json"),indent=4)
    except Exception as e:
        print(e)

    print("Adding checkpoint to app ensemble ...")
    try:
        if args.add_to_ensemble:
            ensemble = []
            with open("./server/ensemble.sh", "r") as f:
                ensemble = [line.strip() for line in f]
            if experiment not in ensemble:
                ensemble.append(experiment)
            with open("./server/ensemble.sh", "w") as f:
                for line in ensemble:
                    f.write(line+"\n")
    except Exception as e:
        print(e)
    
    print("Downloading code ...")
    try:
        if args.back_sync:
            RSYNC(pjoin(args.server, "Sound2Synth"),"../",args="-r -vP -q --exclude '*.wav'")
    except Exception as e:
        print(e)
