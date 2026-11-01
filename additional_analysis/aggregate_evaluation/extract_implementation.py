"""Extract checkpoint metadata, layer tensor shapes, and logged timing evidence."""
from pathlib import Path
import argparse
import ast
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'engine'))
import torch
from checkpoint_reader import load_checkpoint
from common import save_json,file_identity,write_csv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='Data/new_pipeline_512_20000a')
    p.add_argument('--output',default='Data/initial_refinement_results/implementation')
    a=p.parse_args();train=Path(a.root)/'training_v28';out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    files={'inverse':'reconstruction_model_supervised_roi_tof_best.pth','autoencoder':'latent_sos_autoencoder_v1.pth',
           'forward_operator':'matched_tof_operator_v1.pth','latent_refiner':'adjoint_latent_background_clean.pt',
           'spatial_refiner':'spatial_physics_reconstructor_clean.pt'}
    records={};layers=[]
    for name,filename in files.items():
        path=train/filename
        if not path.exists():raise FileNotFoundError(path)
        ck=load_checkpoint(path)
        metadata={}
        for key,value in ck.items():
            if key in ('config','args','train_config','epoch','best_epoch','best_score','best_val_loss','kind','version','in_ch'):
                metadata[key]=value
        states={}
        for key,value in ck.items():
            if isinstance(value,dict) and value and all(torch.is_tensor(v) for v in value.values()):
                states[key]=dict(tensor_elements=sum(v.numel() for v in value.values()),
                                  floating_tensor_elements=sum(v.numel() for v in value.values() if v.is_floating_point()),
                                  tensors=len(value))
                for k,v in value.items():
                    layers.append(dict(component=name,state=key,tensor=k,shape=list(v.shape),elements=v.numel(),dtype=str(v.dtype)))
        records[name]=dict(file=file_identity(path),metadata=metadata,state_dictionaries=states,
                           note='State tensor counts can include buffers. Exact model parameter counts are emitted during inference.')
        del ck
    save_json(out/'checkpoint_inventory.json',records);write_csv(out/'checkpoint_tensor_shapes.csv',layers)
    source=Path(__file__).resolve().parent/'engine'
    wanted={'model.py':{'DeepGatedReconstructionNet','ResidualRefiner','ConvBlock','UpsampleConvBlock'},
            'latent_operator_models.py':{'ShapeAutoencoder','ShapeDecoder','MatchedToFOperator','ConditionalLatentEncoder','ResidualBlock'},
            'train_background_refiner.py':{'AdjointLatentRefiner'},'train_spatial_physics.py':{'SpatialPhysicsResidualNet','ResidualBlock'}}
    chunks=[]
    for filename,names in wanted.items():
        text=(source/filename).read_text(encoding='utf-8-sig');tree=ast.parse(text)
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name in names:
                chunks.append(f'## {filename}: {node.name}\n\n```python\n{ast.get_source_segment(text,node)}\n```\n')
    (out/'architecture_source.md').write_text('\n'.join(chunks),encoding='utf-8')
    logs=[]
    for folder in sorted(train.glob('myOutputs_stage*')):
        if not folder.is_dir():continue
        for path in sorted(folder.glob('terminal*.txt')):
            text=path.read_text(encoding='utf-8',errors='replace')
            elapsed=[]
            for match in re.finditer(r'\b(?:elapsed(?:_phase)?|TotalElapsed)=([0-9.]+)(h|min|s)',text):
                elapsed.append(float(match[1])*{'h':3600,'min':60,'s':1}[match[2]])
            evidence=[line for line in text.splitlines() if re.search(r'\[CMD\]|GPU|NVIDIA|device name|CUDA Device|Total.*time|Training.*time|Done\.|early stopping|resume|Resume',line)]
            logs.append(dict(file=str(path),last_reported_elapsed_seconds=max(elapsed) if elapsed else None,
                             elapsed_interpretation='logged elapsed for this invocation; not a hardware benchmark or necessarily total training across resumes',
                             evidence=evidence[:100]+(['...'] if len(evidence)>200 else [])+evidence[-100:]))
    save_json(out/'training_log_evidence.json',logs)
    lines=['# Implementation and timing evidence','',
           'Checkpoint configurations and tensor dimensions are extracted from the retained files. Architecture definitions are in architecture_source.md.',
           'Exact instantiated parameter counts and expanded module listings are produced by the inference phase.',
           '','## Logged elapsed times','',
           'These are elapsed observations, not invented total training durations. Resumed runs must be reconciled before reporting a total.','']
    for r in logs:
        if r['last_reported_elapsed_seconds'] is not None:
            lines.append(f'- {Path(r["file"]).parent.name}/{Path(r["file"]).name}: {r["last_reported_elapsed_seconds"]/3600:.3f} h last reported elapsed.')
    lines += ['','## Hardware provenance','',
              'Use training-log hardware evidence for historical training hardware. The validation hardware.json records the machine executing the new evaluation; it must not be presented as historical training hardware without confirmation.',
              'Checkpoint file modification times are not used as training durations.']
    (out/'IMPLEMENTATION_EVIDENCE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Extracted implementation evidence to {out}')


if __name__=='__main__':main()
