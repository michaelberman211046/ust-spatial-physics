"""Summarize saved conditioning, refinement and sensitivity outputs without rerunning models."""
from pathlib import Path
import argparse,csv,json,hashlib


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_rows(path,rows):
    if not rows:return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ablation-results',type=Path,required=True)
    p.add_argument('--extended-results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve()
    for inp in [a.ablation_results.resolve(),a.extended_results.resolve()]:
        if a.output.is_relative_to(inp) or inp.is_relative_to(a.output):raise ValueError('Summary output must be separate from numerical inputs')
    a.output.mkdir(parents=True,exist_ok=True)
    status={};identities={};rows=[]
    agg=read_json(a.ablation_results/'aggregate_metrics.json')
    for name,metrics in agg.items():
        for metric,statistics in metrics.items():rows.append({'model':name,'metric':metric,**statistics})
    write_rows(a.output/'conditioning_metrics.csv',rows)
    for study in ['larger_refinement','support_sensitivity','reconstruction_grid']:
        folder=a.extended_results/study
        cases=[read_json(path) for path in sorted(folder.glob('case_*.json'))]
        planned={'larger_refinement':20,'support_sensitivity':9,'reconstruction_grid':9}[study]
        protocol_path=a.extended_results/'extended_protocol.json'
        if protocol_path.exists():
            cfg=read_json(protocol_path)['studies']
            planned={'larger_refinement':cfg['refinement_samples'],'support_sensitivity':len(cfg['support_conditions']),
                     'reconstruction_grid':(cfg['synthetic_grid_samples']+1)*len(cfg['reconstruction_grids'])}[study]
        status[study]={'planned':planned,'completed':len(cases),'complete':len(cases)==planned}
        write_rows(a.output/(study+'_cases.csv'),cases)
    for study in ['rf_picker','tof_robustness','forward_grid']:
        marker=a.extended_results/study/'complete.json';status[study]={'complete':marker.exists()}
    for parent in [a.ablation_results,a.extended_results]:
        for path in parent.rglob('*'):
            if not path.is_file() or path.suffix not in ['.csv','.json'] or any(k in path.parts for k in ['runtime','logs']):continue
            identities[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    summary=a.extended_results/'larger_refinement/summary.json'
    if summary.exists():
        r=read_json(summary);s=r['summary'];changes={}
        for metric in ['rmse_mps','nrmse_percent','mad_mps','correlation']:
            before=s['initial_'+metric]['mean'];after=s['final_'+metric]['mean']
            changes[metric]={'initial_mean':before,'final_mean':after,'final_minus_initial':after-before,
                'relative_change_percent':100*(after/before-1) if before else None}
        status['paired_refinement_metrics']=changes
    (a.output/'completion_and_changes.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    (a.output/'input_sha256.json').write_text(json.dumps(identities,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
