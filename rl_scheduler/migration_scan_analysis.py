"""Independent offline analysis, reading compact summaries and resource streams only."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import statistics

from .migration_scan import RESOURCE_FIELDS, finite, improvement, weighted_summary
from .real_budget import atomic_json
from .real_bridge import digest


def load(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or ['status'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else 'N/A' if v is None else v
                for k,v in row.items()})


def markdown_table(rows, fields):
    def value(x):
        return 'N/A' if x is None else f'{x:.3f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+
        ['| '+' | '.join(value(row.get(k)) for k in fields)+' |' for row in rows])


def bottleneck(summary):
    resources=summary['resources']
    ids=list(resources['kv_pct']['instances'])
    hot=summary['mapping'][0][0]
    counter=summary['counter_deltas'].get(hot,{})
    coverage=min(resources[k]['instances'][j]['coverage'] for k in ('kv_pct','gpu_pct','waiting') for j in ids)
    wait=resources['waiting']['instances'][hot]['mean']
    spare=any(j!=hot and resources['kv_pct']['instances'][j]['mean'] is not None
        and resources['kv_pct']['instances'][j]['mean']<50
        and resources['gpu_pct']['instances'][j]['mean'] is not None
        and resources['gpu_pct']['instances'][j]['mean']<80 for j in ids)
    kv=counter.get('kv_allocation_later',0)
    seq=counter.get('sequence_limit_blocked',0)
    tokens=counter.get('token_budget_blocked',0)
    persistent=(summary.get('waiting_fraction') or 0)>=.2
    passed=coverage>=.8 and wait is not None and wait>0 and persistent and spare and kv>seq+tokens
    return dict(kv_pressure_supported=passed, hot_instance=hot, minimum_resource_coverage=coverage,
        hot_waiting_mean=wait, waiting_fraction=summary.get('waiting_fraction'),
        spare_instance_observed=spare, observed_kv_blocks=kv,
        observed_sequence_blocks=seq, observed_token_budget_blocks=tokens,
        attribution='observed scheduler check counts; overlapping limits and unsampled intervals remain uncertain',
        reason='KV allocation checks dominate; persistent hot queue and peer headroom' if passed else
            'KV-driven persistent local queue not established, scheduler-limit controls or coverage insufficient')


def recommendation(records):
    """Engineering gate, not a significance test; refuse incomplete evidence."""
    by_policy={}
    for row in records:
        by_policy.setdefault(row['policy'],[]).append(row)
    qualified=[]
    for policy in ('Balanced-L','Balanced-M','Balanced-H'):
        rows=by_policy.get(policy,[])
        if len({r['seed'] for r in rows})<3:
            continue
        beneficial=sum(finite(r.get('goodput_improvement_pct')) and r['goodput_improvement_pct']>=10
            and finite(r.get('error_rate_absolute_change')) and r['error_rate_absolute_change']<=.01 for r in rows)
        evidence=all(r.get('kv_pressure_supported') and r.get('resource_balance_supported')
            and r.get('succeeded',0)>=2 and (r.get('success_rate') or 0)>=.8
            and (r.get('opportunity_ratio') or 0)>0 for r in rows)
        if beneficial>=2 and evidence:
            qualified.append(policy)
    complete=all(len({r['seed'] for r in by_policy.get(p,[])})>=3 for p in ('Balanced-L','Balanced-M','Balanced-H'))
    return dict(recommendation='recommended' if qualified else 'not_recommended' if complete else 'conditional',
        thresholds=qualified, reason='>=10% paired goodput in >=2/3 seeds with KV, migration and balance evidence' if qualified else
        'no threshold passes full evidence gates' if complete else 'paired seeds/thresholds incomplete; no recommendation yet',
        engineering_rules=dict(error_rate_absolute_tolerance=.01,min_successes_per_run=2,min_success_rate=.8))


def analyze(directory):
    directory=Path(directory)
    ledger=directory/'completed_runs.jsonl'
    records=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    latest={}
    for record in records:
        key=digest({k:record[k] for k in ('phase','case','seed','policy','threshold')})
        latest[key]=record
    records=list(latest.values())
    resources, scans, pairs, migrations, migration_stats, balances, performance=[],[],[],[],[],[],[]
    loaded=[]
    resource_table=[]
    candidate_scores=[]
    for record in records:
        summary=load(directory/record['summary'])
        if summary['status']!='passed':
            continue
        loaded.append((record,summary))
        meta=dict(scenario=record['case']['scenario'],seed=record['seed'],policy=record['policy'],threshold=record['threshold'])
        diagnostic=bottleneck(summary)
        metrics=summary['metrics']
        hot=summary['mapping'][0][0]
        for metric,stats in summary['resources'].items():
            for key,value in stats['instances'].items():
                resources.append(dict(**meta,metric=metric,instance_id=key,**value,
                    cluster_mean=stats['cluster_mean'],std_mean=stats['dispersion']['mean'],
                    std_coverage=stats['dispersion']['coverage'],spread_mean=stats['spread']['mean']))
        for m in summary['migrations']:
            migrations.append(dict(**meta,**m))
        migration_stats.append(dict(**meta,**summary['migration_statistics']))
        table=dict(**meta,**{k:summary['resources'][k]['cluster_mean'] for k in
            ('kv_pct','gpu_pct','memory_mib','memory_pct','running','waiting','output_tps')},
            hot_kv_p95=summary['resources']['kv_pct']['instances'][hot]['p95'],
            hot_gpu_mean=summary['resources']['gpu_pct']['instances'][hot]['mean'],
            kv_std=summary['resources']['kv_pct']['dispersion']['mean'],
            gpu_std=summary['resources']['gpu_pct']['dispersion']['mean'],
            waiting_p95=max((v['p95'] for v in summary['resources']['waiting']['instances'].values() if v['p95'] is not None),default=None))
        resource_table.append(table)
        row=dict(**meta,**metrics,**diagnostic,summary=record['summary'])
        if record['phase']=='scan':
            scans.append(row)
            if diagnostic['kv_pressure_supported']:
                candidate_scores.append((diagnostic['hot_waiting_mean'],record['case']))
        else:
            pairs.append(row)
    csv_write(directory/'stay_scan.csv',scans)
    csv_write(directory/'paired_experiments.csv',pairs)
    csv_write(directory/'migration_events.csv',migrations)
    csv_write(directory/'migration_statistics.csv',migration_stats)
    csv_write(directory/'resource_utilization.csv',resources)
    csv_write(directory/'resource_summary_table.csv',resource_table)
    candidates=[]
    for _,case in sorted(candidate_scores,key=lambda x:x[0],reverse=True):
        if 'control_of' not in case and case not in candidates and len(candidates)<3:
            candidates.append(case)
    atomic_json(directory/'candidate_shortlist.json',candidates)
    controls={}
    for record,summary in loaded:
        parent=record['case'].get('control_of')
        if parent:
            controls.setdefault(parent,[]).append((record['case'],bottleneck(summary)))
    def controls_passed(scenario):
        measured=controls.get(scenario,[])
        return any(c['max_num_seqs']==256 and d['kv_pressure_supported'] for c,d in measured) and any(
            c['max_num_batched_tokens']==2112 and d['kv_pressure_supported'] for c,d in measured)
    comparison_rows=[]
    for record,summary in loaded:
        if record['phase']!='paired' or record['policy']=='Stay':
            continue
        matches=[s for r,s in loaded if r['phase']=='paired' and r['policy']=='Stay'
            and r['case']==record['case'] and r['seed']==record['seed']]
        if len(matches)!=1:
            continue
        stay=matches[0]
        if stay['trace_hash']!=summary['trace_hash'] or stay['mapping']!=summary['mapping']:
            raise ValueError('unpaired trajectory or instance epochs')
        meta=dict(scenario=record['case']['scenario'],seed=record['seed'],policy=record['policy'],threshold=record['threshold'])
        row=dict(**meta,**summary['migration_statistics'],kv_pressure_supported=bottleneck(stay)['kv_pressure_supported'] and controls_passed(record['case']['scenario']))
        reduced=[]
        for metric in ('kv_pct','gpu_pct','waiting','running'):
            a=stay['resources'][metric]['dispersion']['mean']; b=summary['resources'][metric]['dispersion']['mean']
            diff=improvement(a,b,lower=True)
            balances.append(dict(**meta,metric=metric,stay_std=a,balanced_std=b,**diff))
            if diff['absolute'] is not None:
                reduced.append(diff['absolute']>0)
        hot=stay['mapping'][0][0]
        row['resource_balance_supported']=any(reduced) and (
            finite(summary['resources']['waiting']['instances'][hot]['mean']) and
            finite(stay['resources']['waiting']['instances'][hot]['mean']) and
            summary['resources']['waiting']['instances'][hot]['mean'] < stay['resources']['waiting']['instances'][hot]['mean'])
        for metric,lower in [('slo_goodput_rps',False),('ttft_p95_ms',True),('tpot_p95_ms',True),
            ('completed_throughput_rps',False),('client_token_throughput_tps',False),('e2e_mean_ms',True),('e2e_p95_ms',True),
            ('nonmanual_error_rate',False)]:
            a=stay['metrics'].get(metric);b=summary['metrics'].get(metric)
            diff=improvement(a,b,lower=lower)
            performance.append(dict(**meta,metric=metric,stay=a,balanced=b,**diff))
            if metric=='slo_goodput_rps': row['goodput_improvement_pct']=diff['percent']
            if metric=='ttft_p95_ms': row['ttft_improvement_pct']=diff['percent']
            if metric=='tpot_p95_ms': row['tpot_improvement_pct']=diff['percent']
            if metric=='nonmanual_error_rate': row['error_rate_absolute_change']=diff['absolute']
        comparison_rows.append(row)
    # Include Stay in the final four-policy table, retaining each seed.
    for record,summary in loaded:
        if record['phase']=='paired' and record['policy']=='Stay':
            comparison_rows.append(dict(scenario=record['case']['scenario'],seed=record['seed'],policy='Stay',threshold=None,
                **summary['migration_statistics'], goodput_improvement_pct=improvement(summary['metrics']['slo_goodput_rps'],
                    summary['metrics']['slo_goodput_rps'])['percent'],ttft_improvement_pct=None,tpot_improvement_pct=None))
    csv_write(directory/'load_balance_comparison.csv',balances)
    csv_write(directory/'performance_improvement.csv',performance)
    csv_write(directory/'migration_performance_table.csv',comparison_rows)
    aggregate=[]
    for scenario,policy in sorted({(r['scenario'],r['policy']) for r in comparison_rows}):
        group=[r for r in comparison_rows if (r['scenario'],r['policy'])==(scenario,policy)]
        for metric in ('goodput_improvement_pct','ttft_improvement_pct','tpot_improvement_pct','succeeded','request_coverage','opportunity_ratio'):
            values=[r[metric] for r in group if finite(r.get(metric))]
            aggregate.append(dict(scenario=scenario,policy=policy,metric=metric,n=len(values),
                mean=statistics.mean(values) if values else None,std=statistics.pstdev(values) if values else None))
    for scenario,policy in sorted({(r['scenario'],r['policy']) for r in resource_table}):
        group=[r for r in resource_table if (r['scenario'],r['policy'])==(scenario,policy)]
        for metric in ('kv_pct','gpu_pct','memory_mib','memory_pct','kv_std','gpu_std','waiting_p95','output_tps'):
            values=[r[metric] for r in group if finite(r.get(metric))]
            aggregate.append(dict(scenario=scenario,policy=policy,metric=metric,n=len(values),
                mean=statistics.mean(values) if values else None,std=statistics.pstdev(values) if values else None))
    csv_write(directory/'cross_seed_statistics.csv',aggregate)
    recommendations={scenario:recommendation([r for r in comparison_rows if r['scenario']==scenario])
        for scenario in {r['scenario'] for r in scans+pairs}}
    for row in comparison_rows:
        row['recommendation']=recommendations[row['scenario']]['recommendation']
    csv_write(directory/'migration_performance_table.csv',comparison_rows)
    atomic_json(directory/'recommendations.json',recommendations)
    report='真实完成的 Stay 组数：'+str(len(scans))+'；配对组数：'+str(len(pairs))+'。\n\n'
    report+='表中 Waiting P95 为四实例各自时间加权 P95 的最大值。资源表含每卡覆盖率、均值、P95、极值及实例间标准差。\n\n无真实记录时不提供实测值。缺失数据为 N/A；覆盖率单独报告。均衡不等于业务收益。\n\n'
    report+=markdown_table(resource_table,['scenario','seed','policy','threshold','kv_pct','gpu_pct','memory_mib','hot_kv_p95','kv_std','gpu_std','waiting_p95'])
    (directory/'stay_scan_report.md').write_text(report+'\n')
    (directory/'resource_bottleneck_analysis.md').write_text(
        '阻塞计数来自实际 can_allocate/can_schedule；计数可重叠，不表示互斥持续时间。\n'
        'KV 与 Waiting 持续性、同伴 GPU 余量须同时满足；128/256 seq 与 2112 token 控制用于排除人工瓶颈。\n'
        '显存预分配占用不等于 KV 使用量。抢占仅报告实际模式，不把未知模式强称为 KV 原因。\n\n'+
        markdown_table(scans,['scenario','kv_pressure_supported','waiting_fraction','minimum_resource_coverage',
            'observed_kv_blocks','observed_sequence_blocks','observed_token_budget_blocks','reason'])+'\n')
    (directory/'ppo_scenario_recommendations.md').write_text(report+'\n\n'+
        markdown_table(comparison_rows,['scenario','seed','policy','threshold','succeeded','successes_per_minute',
            'request_coverage','opportunity_ratio','action_given_opportunity','goodput_improvement_pct',
            'ttft_improvement_pct','tpot_improvement_pct','recommendation'])+'\n\n'+json.dumps(recommendations,ensure_ascii=False,indent=2)+
        '\n\n调度上限对照尚未确认 KV 压力时，不予 recommended。无候选时先提高实际输出/输入 KV 需求，逐级扩展 QPS；必要时单独评估显存预算，禁止压低并发制造排队。\n探索性工程门槛，不作统计显著性结论。资料不全为 conditional，未通过不得作为正向预训练环境。\n')
    if loaded:
        plots(directory,loaded,resource_table,balances,comparison_rows)


def plots(directory,loaded,resource_rows,balances,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[('kv_utilization_timeseries.png',['kv_pct']),('gpu_utilization_timeseries.png',['gpu_pct']),
        ('gpu_memory_timeseries.png',['memory_mib']),('running_waiting_timeseries.png',['running','waiting'])]
    for filename,fields in specs:
        fig,ax=plt.subplots(figsize=(12,5))
        for record,summary in loaded:
            path=directory/record['summary']
            # Read only newly recorded compact resource streams, never events/history.
            samples=[json.loads(s) for s in (path.parent/'resources.jsonl').read_text().splitlines()]
            for key in [m[0] for m in summary['mapping']]:
                for field in fields:
                    xs=[];ys=[]
                    for sample in samples:
                        rows={r['instance_id']:r for r in sample['instances']}
                        xs.append(sample['at']-summary['window']['start'])
                        value=rows.get(key,{}).get(field)
                        ys.append(value if value is not None else float('nan'))
                    ax.plot(xs,ys,label=record['case']['scenario']+'/'+record['policy']+'/'+str(record['seed'])+'/'+key[:5]+'/'+field,alpha=.6)
        ax.set_xlabel('seconds from measured start');ax.set_ylabel(', '.join(fields))
        if len(loaded)<=2:ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(directory/'plots'/filename);plt.close(fig)
    for filename,rows,fields in [('resource_utilization_comparison.png',resource_rows,['kv_pct','gpu_pct','memory_mib','memory_pct']),
        ('load_balance_improvement.png',balances,['percent']),('migration_frequency_comparison.png',comparisons,['successes_per_minute']),
        ('performance_improvement.png',comparisons,['goodput_improvement_pct','ttft_improvement_pct','tpot_improvement_pct'])]:
        fig,axes=plt.subplots(len(fields),1,figsize=(12,4*len(fields)),squeeze=False)
        xs=[r['scenario']+'/'+r['policy']+'/'+str(r['seed'])+('/'+r['metric'] if 'metric' in r else '') for r in rows]
        for ax,field in zip(axes.flat,fields):
            ys=[r.get(field) if finite(r.get(field)) else float('nan') for r in rows]
            ax.bar(range(len(xs)),ys);ax.set_xticks(range(len(xs)),xs,rotation=90,fontsize=5)
            ax.set_ylabel(field)
        fig.tight_layout();fig.savefig(directory/'plots'/filename);plt.close(fig)
