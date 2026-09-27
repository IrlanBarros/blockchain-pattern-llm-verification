from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from llm_pipeline.aggregation import aggregate_issue_results
from llm_pipeline.checkpoint import resumable_batch
from llm_pipeline.client import call_sync, parse_response_payload
from llm_pipeline.compact_wire import CANDIDATE, S1, S2, decode_payload, wire_schema
from llm_pipeline.context import prepare_optimized, select_context, select_spans
from llm_pipeline.data import load_patterns, prepare_issue
from llm_pipeline.optimization import OptimizationConfig
from llm_pipeline.optimization_runtime import save_optimization_manifest, validate_optimization_resume
from llm_pipeline.requests import stage1_request_params, stage2_request_params
from llm_pipeline.retrieval import CompactCatalog, HybridRetriever, LatentSemanticIndex
from llm_pipeline.schemas import normalize_stage1, normalize_stage2, stage1_schema, stage2_schema
from llm_pipeline.stages import run_stage1_sync, run_stage2_sync, stage2_pairs_from_stage1, pipeline_integrity_report
from llm_pipeline.telemetry import estimate_tokens, observed_sync, read_calls, summarize_calls
from test_gemini_stages import FakeSyncClient, response_for, stage1_json, stage2_json

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def catalog():
    return load_patterns(ROOT/'blockchain_patterns_keywords_v3.csv')


@pytest.fixture(scope='module')
def retriever(catalog):
    return HybridRetriever(catalog,OptimizationConfig(enabled=True))


def row(repository='r', number='1', body='Fetch price off-chain and submit it to the contract.', comments=''):
    return pd.Series({'repository':repository,'issue_number':number,'issue_title':'External price feed',
                      'issue_body':body,'concatenated_comments':comments,'type':'Issue','state':'open','labels':''})


def compact_payload(text, catalog, stage):
    payload=json.loads(text)
    compact=CompactCatalog(catalog)
    if stage==1:
        payload['candidates']=[{CANDIDATE[k]:compact.records[v].pattern_id if k=='pattern' else v
                                for k,v in c.items()} for c in payload['candidates']]
        return json.dumps({S1[k]:v for k,v in payload.items()})
    for k in ('mechanism_match','scope_match','focus_match'):
        payload[k]=True
    return json.dumps({S2[k]:v for k,v in payload.items()})


def test_compact_catalog_complete_and_ids_stable(catalog):
    compact=CompactCatalog(catalog)
    assert len(compact.records)==82
    assert compact.id_to_name['P002']=='Oracle'
    assert all(r.short_definition and r.distinctive_mechanisms and r.category for r in compact.records.values())
    reverse=CompactCatalog(replace(catalog,names=list(reversed(catalog.names))))
    assert reverse.id_to_name==compact.id_to_name
    for n,r in compact.records.items():
        assert compact.canonicalize(r.pattern_id)==n
    with pytest.raises(ValueError,match='Unknown'):
        compact.canonicalize('P999')


def test_changed_source_definition_never_uses_stale_summary(catalog):
    by_name={**catalog.by_name,'Oracle':{**catalog.by_name['Oracle'],'description':'Changed definition.'}}
    assert CompactCatalog(replace(catalog,by_name=by_name)).records['Oracle'].short_definition=='Changed definition.'


def test_new_catalog_names_do_not_shift_existing_ids(catalog):
    new={'pattern':'New pattern','description':'Novel mechanism.','category':'New','subcategory':'New'}
    updated=replace(catalog,names=['New pattern']+catalog.names,by_name={**catalog.by_name,'New pattern':new},
                    dataframe=pd.concat([catalog.dataframe,pd.DataFrame([new])],ignore_index=True))
    compact=CompactCatalog(updated)
    assert compact.records['Oracle'].pattern_id=='P002'
    assert compact.records['New pattern'].pattern_id.startswith('PX')


def test_lexical_union_never_hard_capped(catalog):
    config=OptimizationConfig(enabled=True,retrieval_min_candidates=1,retrieval_max_candidates=2,
                 full_catalog_fallback=False,retrieval_family_expansion=False)
    r=HybridRetriever(catalog,config)
    r.semantic.scores=lambda _: [0.] * len(r.names)
    names=['Oracle','Proxy contract','Ownable','Mutex','Snapshotting']
    result=r.retrieve(' '.join(names))
    assert set(names)<=set(result['retrieval_candidates'])
    assert result['retrieval_candidate_count']>config.retrieval_max_candidates


def test_lexical_alias_and_keyword_not_discarded(catalog):
    r=HybridRetriever(catalog,OptimizationConfig(enabled=True,full_catalog_fallback=False))
    r.semantic.scores=lambda _: [0.] * len(r.names)
    result=r.retrieve('Use an upgradeable proxy and latestRoundData().')
    assert {'Proxy contract','Ticker tape'}<=set(result['retrieval_candidates'])
    assert result['retrieval_scores']['Proxy contract']['direct_match']


def test_semantic_paraphrase_without_pattern_name(catalog):
    r=HybridRetriever(catalog,OptimizationConfig(enabled=True,full_catalog_fallback=False,
                retrieval_require_lexical_anchor=False,retrieval_confidence_threshold=0))
    result=r.retrieve('Forward function calls to a replaceable implementation while retaining the public address.')
    assert 'Proxy contract' in result['retrieval_candidates']
    assert result['retrieval_scores']['Proxy contract']['semantic']>0
    assert not result['retrieval_scores']['Proxy contract']['direct_match']


def test_semantic_index_reproducible_and_empty():
    docs=['external weather readings enter blockchain','contract delegates calls replaceable implementation','threshold key shares reconstruct secret']
    first=LatentSemanticIndex(docs,2)
    second=LatentSemanticIndex(docs,2)
    assert first.scores('weather readings')==second.scores('weather readings')
    assert first.scores('zzzzzz')==[0,0,0]


def test_family_expansion_preserves_confusable_neighbors(retriever):
    names=retriever.retrieve('An upgradeable proxy uses delegatecall.')['retrieval_candidates']
    assert {'Proxy contract','Contract Registry (on-chain)','Router contract','Data contract'}<=set(names)


@pytest.mark.parametrize('config,text',[
    (OptimizationConfig(enabled=True),'zzzzzzzz qqqqqqqq'),
    (OptimizationConfig(enabled=True,force_full_catalog=True),'Oracle'),
    (OptimizationConfig(enabled=True,retrieval_enabled=False),'Oracle'),
])
def test_safe_full_catalog_fallback(catalog,config,text):
    result=HybridRetriever(catalog,config).retrieve(text)
    assert result['retrieval_candidate_count']==82
    assert result['retrieval_fallback_used']


def test_unanchored_lsa_uses_full_catalog_by_default(retriever):
    result=retriever.retrieve('Weather measurements submitted externally.')
    assert 'no_lexical_anchor' in result['retrieval_fallback_reasons']
    assert result['retrieval_candidate_count']==82


def test_selective_false_friends(catalog):
    text=CompactCatalog(catalog).false_friends_context(['Oracle'])
    assert 'Oracle Database' in text
    assert 'nginx' not in text and 'snapshot' not in text


def test_context_selects_middle_comment_and_preserves_original_offsets(retriever):
    target='Contract updates balances before external calls, preventing reentrancy.'
    comments=('Unrelated continuous integration status.\n'*300)+target+'\n'+('Meeting agenda and boilerplate.\n'*300)
    p=prepare_optimized(row(body='Fix a recursive contract withdrawal.',comments=comments),12000,retriever)
    chosen=select_context(p,retriever,['Check-Effects-Interactions'],stage=2,required=(target,))
    assert target in chosen.artifact_text
    assert chosen.input_truncated
    assert '<repository>' not in chosen.artifact_text and '<issue_number>' not in chosen.artifact_text
    for a,b in chosen.selected_spans['comments']:
        assert comments[a:b] in chosen.artifact_text
    assert chosen.artifact_text.count('External price feed')==1
    assert chosen.repository=='r' and chosen.issue_number=='1'
    assert len(chosen.artifact_text)<=retriever.config.stage2_max_context_tokens*4


def test_context_reports_omitted_comments_and_retains_body(retriever):
    p=prepare_optimized(row(comments='unrelated build output\n'*1500),12000,retriever)
    assert row().issue_body in p.artifact_text
    assert '<context_omitted>' in p.artifact_text
    assert p.input_truncated


def test_required_evidence_budget_fails_explicitly():
    with pytest.raises(ValueError,match='Stage 1 evidence'):
        select_spans('a'*1000,100,'',100,required=('a'*1000,))


def test_wire_roundtrip_keeps_all_stage1_fields(catalog):
    encoded=json.loads(compact_payload(stage1_json(),catalog,1))
    actual=normalize_stage1(decode_payload(encoded,catalog,stage=1,allowed=['Oracle']),catalog)
    assert actual==normalize_stage1(json.loads(stage1_json()),catalog)
    with pytest.raises(ValueError,match='shortlist'):
        decode_payload(encoded,catalog,stage=1,allowed=['Mutex'])


def test_wire_roundtrip_keeps_stage2_flags_and_annotations(catalog):
    encoded=json.loads(compact_payload(stage2_json(),catalog,2))
    encoded.update(a='P001',o=['P002'])
    decoded=decode_payload(encoded,catalog,stage=2)
    assert decoded['alternative_pattern']=='Ticker tape' and decoded['overlap_with']==['Oracle']
    normalized=normalize_stage2(decoded,catalog)
    assert normalized['verdict']=='yes' and normalized['evidence_text'] and normalized['justification']
    encoded['m']='false'
    with pytest.raises(ValueError,match='boolean'):
        decode_payload(encoded,catalog,stage=2)


@pytest.mark.parametrize('invalid',[{'c':['P002']},{'c':[],'candidates':[]},{'v':'yes'}])
def test_compact_incomplete_or_mixed_json_is_rejected(catalog,invalid):
    with pytest.raises(ValueError):
        decode_payload(invalid,catalog,stage=2 if 'v' in invalid else 1)


def test_optimized_requests_have_stable_prefix_and_single_full_description(catalog,retriever):
    p=prepare_optimized(row(),12000,retriever)
    other=prepare_optimized(row(number='2',body='Proxy contract uses delegatecall.'),12000,retriever)
    a=stage1_request_params(p,catalog,'model',1.,2048,0,'minimal')
    b=stage1_request_params(other,catalog,'model',1.,2048,0,'minimal')
    assert a['config']['system_instruction']==b['config']['system_instruction']
    assert a['config']['response_json_schema']['properties']['c']['items']['properties']['p']['enum']
    s2=stage2_request_params(p,'Oracle',catalog,'model',1.,2048,0,'low')
    assert s2['contents'][0]['parts'][0]['text'].count(catalog.by_name['Oracle']['description'])==1
    assert 'nginx proxy' not in s2['contents'][0]['parts'][0]['text']
    from google.genai import types
    types.GenerateContentConfig(**s2['config'])
    for field in S2.values():
        assert field in s2['config']['response_json_schema']['required']


def test_tokens_and_malformed_response_usage_retained(tmp_path,retriever,catalog):
    assert estimate_tokens('')==0 and estimate_tokens('12345')==2
    p=prepare_optimized(row(),12000,retriever)
    client=FakeSyncClient([response_for('not json','model')])
    frame=run_stage1_sync(client,[p],catalog,'model',1.,2048,0,'minimal',tmp_path)
    assert frame.loc[0,'request_status']=='errored'
    rows=read_calls(tmp_path)
    assert len(rows)==1 and rows[0]['input_tokens']==100 and rows[0]['total_tokens']==125
    assert rows[0]['prompt_chars']>1000 and rows[0]['elapsed_seconds']>=0
    summary=summarize_calls(tmp_path,retriever.config)
    assert summary['total_tokens']==125 and summary['total_thoughts_tokens']==5


def test_retry_attempts_and_thinking_accounted(monkeypatch,tmp_path,retriever,catalog):
    p=prepare_optimized(row(),12000,retriever)
    calls=[]
    def generate(**params):
        calls.append(params)
        if len(calls)==1:
            raise RuntimeError('network temporarily unavailable')
        return response_for(stage1_json(),'model')
    client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr('llm_pipeline.client.time.sleep',lambda _:None)
    params=stage1_request_params(p,catalog,'model',1.,2048,0,'minimal')
    observed_sync(call_sync,client,params,tmp_path,p,'stage1')
    rows=read_calls(tmp_path)
    assert len(rows)==2 and rows[0]['input_tokens'] is None
    summary=summarize_calls(tmp_path,retriever.config)
    assert summary['total_tokens']==125 and not summary['usage_complete']
    assert summary['estimated_cost_if_api'] is None


def test_resume_manifest_rejects_config_and_selection_changes(tmp_path,retriever,catalog):
    p=prepare_optimized(row(),12000,retriever)
    save_optimization_manifest(tmp_path,retriever.config,catalog,[p])
    save_optimization_manifest(tmp_path,retriever.config,catalog,[p])
    validate_optimization_resume(tmp_path,retriever.config,catalog)
    with pytest.raises(ValueError,match='changed'):
        validate_optimization_resume(tmp_path,replace(retriever.config,retrieval_max_candidates=11),catalog)
    with pytest.raises(ValueError,match='differs'):
        save_optimization_manifest(tmp_path,retriever.config,catalog,[replace(p,artifact_text='changed')])


def test_legacy_run_cannot_silently_enable_retrieval(tmp_path,catalog):
    with pytest.raises(ValueError,match='legacy'):
        validate_optimization_resume(tmp_path,OptimizationConfig(enabled=True),catalog)


def test_optimized_full_flow_composite_keys_partial_failure_and_resume(tmp_path,retriever,catalog):
    items=[prepare_optimized(row(repository=r),12000,retriever) for r in ['repo/a','repo/b']]
    client=FakeSyncClient([response_for(compact_payload(stage1_json(),catalog,1),'model'),response_for('bad JSON','model')])
    first=run_stage1_sync(client,items,catalog,'model',1.,2048,0,'minimal',tmp_path)
    assert len(first)==2 and (first.request_status=='succeeded').sum()==1
    retry=FakeSyncClient([response_for(compact_payload(stage1_json(),catalog,1),'model')])
    s1=run_stage1_sync(retry,items,catalog,'model',1.,2048,0,'minimal',tmp_path)
    assert len(retry.models.calls)==1 and len(s1)==2
    assert all(json.loads(v)==['Oracle'] for v in s1.candidates)
    pairs=stage2_pairs_from_stage1(s1)
    verifier=FakeSyncClient([response_for(compact_payload(stage2_json(),catalog,2),'model') for _ in pairs])
    s2=run_stage2_sync(verifier,{(p.repository,p.issue_number):p for p in items},pairs,catalog,'model',1.,2048,0,'low',tmp_path)
    assert len(s2)==2 and set(s2.pattern)=={'Oracle'}
    assert pipeline_integrity_report(items,s1,s2)['status']=='OK'
    assert len(aggregate_issue_results(s1,s2))==2
    unchanged=(tmp_path/'stage2_results.csv').read_bytes()
    s2again=run_stage2_sync(None,{(p.repository,p.issue_number):p for p in items},pairs,catalog,'model',1.,2048,0,'low',tmp_path)
    assert len(s2again)==2 and (tmp_path/'stage2_results.csv').read_bytes()==unchanged
    assert len(read_calls(tmp_path))==5


def test_remote_batch_resume_does_not_resubmit_after_interrupt(tmp_path):
    calls=[]
    def create(client,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name='batches/persisted')
    def interrupted(*args):
        raise KeyboardInterrupt()
    requests=[{'metadata':{'custom_id':'one'},'contents':[]}]
    params=dict(model='m',requests=requests,display_name='test',run_dir=tmp_path,stage='stage1',create=create,poll_seconds=1)
    with pytest.raises(KeyboardInterrupt):
        resumable_batch(None,wait=interrupted,**params)
    name,created,final=resumable_batch(None,wait=lambda *a:'complete',**params)
    assert name=='batches/persisted' and final=='complete' and len(calls)==1


def test_config_environment_and_invalid_values(monkeypatch):
    monkeypatch.setenv('RETRIEVAL_MAX_CANDIDATES','17')
    monkeypatch.setenv('FULL_CATALOG_FALLBACK','false')
    cfg=OptimizationConfig.from_environment(enabled=True)
    assert cfg.retrieval_max_candidates==17 and not cfg.full_catalog_fallback
    with pytest.raises(ValueError):
        OptimizationConfig(retrieval_min_candidates=20,retrieval_max_candidates=2)
    monkeypatch.setenv('FULL_CATALOG_FALLBACK','maybe')
    with pytest.raises(ValueError):
        OptimizationConfig.from_environment()


def test_benchmark_missing_labels_are_unknown_not_perfect(catalog,retriever):
    from benchmark_tokens import retrieval_metrics
    p=prepare_optimized(row(),12000,retriever)
    frame=pd.DataFrame([row()])
    report=retrieval_metrics(frame,[p],catalog)
    assert report['recall'] is None and report['missed_positive_count'] is None
    frame['relevant_to_pattern_study']='yes'
    frame['patterns_present']='Oracle'
    report=retrieval_metrics(frame,[p],catalog)
    assert report['human_positive_pairs']==1 and report['recall']==1
    changed=replace(p,retrieval={**p.retrieval,'retrieval_candidates':[]})
    report=retrieval_metrics(frame,[changed],catalog)
    assert report['recall']==0 and report['missed_positive_count']==1


def test_compact_mode_false_preserves_verbose_response_schema(catalog,retriever):
    p=prepare_optimized(row(),12000,retriever)
    p=replace(p,optimization=replace(p.optimization,compact_output=False))
    params=stage1_request_params(p,catalog,'model',1,2048,0,'minimal')
    assert 'candidates' in params['config']['response_json_schema']['properties']


def test_optimized_batch_end_to_end_and_usage(tmp_path,retriever,catalog):
    from test_gemini_stages import FakeBatchClient
    from llm_pipeline.stages import run_stage1_batch, run_stage2_batch
    from llm_pipeline.utils import stable_custom_id
    p=prepare_optimized(row(),12000,retriever)
    inline=SimpleNamespace(metadata={'custom_id':p.custom_id_stage1},error=None,
                           response=response_for(compact_payload(stage1_json(),catalog,1),'model'))
    client=FakeBatchClient(inline)
    s1=run_stage1_batch(client,[p],catalog,'model',1,2048,0,'minimal',10,1000000,0,tmp_path)
    assert json.loads(s1.iloc[0].candidates)==['Oracle']
    pair=stable_custom_id('s2',p.repository,p.issue_number,'Oracle')
    inline=SimpleNamespace(metadata={'custom_id':pair},error=None,
                           response=response_for(compact_payload(stage2_json(),catalog,2),'model'))
    s2=run_stage2_batch(FakeBatchClient(inline),{(p.repository,p.issue_number):p},stage2_pairs_from_stage1(s1),
                        catalog,'model',1,2048,0,'low',10,1000000,0,tmp_path)
    assert s2.iloc[0].verdict=='yes'
    calls=read_calls(tmp_path)
    assert len(calls)==2 and calls[0]['candidate_count']==1
    assert all(c['total_tokens']==125 and c['elapsed_seconds'] is None for c in calls)
    assert summarize_calls(tmp_path,retriever.config)['total_tokens']==250


def test_cli_optimized_pipeline_then_resume_without_client(tmp_path,monkeypatch,catalog):
    from llm_pipeline.cli import main
    input_path=tmp_path/'input.csv'
    pd.DataFrame([row()]).to_csv(input_path,index=False)
    client=FakeSyncClient([response_for(compact_payload(stage1_json(),catalog,1),'model'),
                           response_for(compact_payload(stage2_json(),catalog,2),'model')])
    monkeypatch.setattr('llm_pipeline.cli.create_client',lambda:client)
    args=['run','--input',str(input_path),'--patterns',str(ROOT/'blockchain_patterns_keywords_v3.csv'),
          '--profile','optimized','--output-dir',str(tmp_path),'--run-id','live']
    assert main(args+['--stage','stage1'])==0
    assert len(client.models.calls)==1
    assert main(args+['--stage','stage2'])==0
    assert len(client.models.calls)==2
    def no_client():
        pytest.fail('Completed resume must not create an API client')
    monkeypatch.setattr('llm_pipeline.cli.create_client',no_client)
    assert main(args)==0
    run=tmp_path/'live'
    assert json.loads((run/'integrity_report.json').read_text())['status']=='OK'
    assert json.loads((run/'token_summary.json').read_text())['calls']==2
    assert (run/'optimization_manifest.json').exists() and (run/'compact_catalog.json').exists()


def test_dry_run_resume_does_not_duplicate_requests(tmp_path):
    from llm_pipeline.cli import main
    input_path=tmp_path/'input.csv'
    pd.DataFrame([row()]).to_csv(input_path,index=False)
    args=['run','--input',str(input_path),'--patterns',str(ROOT/'blockchain_patterns_keywords_v3.csv'),
          '--profile','optimized','--mode','dry-run','--output-dir',str(tmp_path),'--run-id','dry']
    assert main(args)==0
    content=(tmp_path/'dry'/'stage1_requests.jsonl').read_bytes()
    assert main(args)==0
    assert (tmp_path/'dry'/'stage1_requests.jsonl').read_bytes()==content


def test_live_comparison_refuses_to_approve_without_human_labels(tmp_path,monkeypatch,catalog):
    from llm_pipeline.cli import main
    from compare_pipeline_runs import compare
    input_path=tmp_path/'input.csv'
    pd.DataFrame([row()]).to_csv(input_path,index=False)
    responses=[response_for(stage1_json(),'model'),response_for(stage2_json(),'model')]
    monkeypatch.setattr('llm_pipeline.cli.create_client',lambda:FakeSyncClient(responses))
    common=['run','--input',str(input_path),'--patterns',str(ROOT/'blockchain_patterns_keywords_v3.csv'),
            '--output-dir',str(tmp_path)]
    assert main(common+['--run-id','baseline'])==0
    responses=[response_for(compact_payload(stage1_json(),catalog,1),'model'),
               response_for(compact_payload(stage2_json(),catalog,2),'model')]
    assert main(common+['--run-id','optimized','--profile','optimized'])==0
    args=SimpleNamespace(baseline_run=str(tmp_path/'baseline'),optimized_run=str(tmp_path/'optimized'),
                         human_issues=None,human_pairs=None,output=str(tmp_path/'comparison.json'))
    assert compare(args)==2
    report=json.loads((tmp_path/'comparison.json').read_text())
    assert report['acceptance']=='NOT_APPROVED'
    assert report['percentage_saved_vs_baseline']==0  # Same mocked usage, never an estimate.
    assert report['runs']['optimized']['retrieval_recall'] is None
    assert report['new_stage1_false_negatives'] is None


def test_frozen_legacy_prompts_unchanged():
    from llm_pipeline.prompts import COMMON_METHOD_RULES,STAGE1_RULES,STAGE2_RULES
    frozen=json.loads((ROOT/'benchmarks/token_optimization/baseline_prompts.json').read_text())
    assert frozen=={'common':COMMON_METHOD_RULES,'stage1':STAGE1_RULES,'stage2':STAGE2_RULES}


def test_batch_resume_subset_preserves_original_response_order(tmp_path):
    requests=[{'metadata':{'custom_id':k},'contents':[]} for k in ['z','a','m']]
    calls=[]
    def create(client,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name='batch/order')
    def interrupted(*args):
        raise KeyboardInterrupt()
    kwargs=dict(model='m',display_name='test',run_dir=tmp_path,stage='stage1',create=create,poll_seconds=0)
    with pytest.raises(KeyboardInterrupt):
        resumable_batch(None,requests=requests,wait=interrupted,**kwargs)
    _,created,_=resumable_batch(None,requests=requests[1:2],wait=lambda *a:None,**kwargs)
    assert created['_request_order']==['z','a','m'] and len(calls)==1


def test_telemetry_can_be_disabled_in_legacy_mode(tmp_path,monkeypatch):
    from llm_pipeline.cli import main
    input_path=tmp_path/'input.csv'
    pd.DataFrame([row()]).to_csv(input_path,index=False)
    monkeypatch.setenv('TOKEN_TELEMETRY_ENABLED','false')
    client=FakeSyncClient([response_for(stage1_json(),'model'),response_for(stage2_json(),'model')])
    monkeypatch.setattr('llm_pipeline.cli.create_client',lambda:client)
    assert main(['run','--input',str(input_path),'--patterns',str(ROOT/'blockchain_patterns_keywords_v3.csv'),
                 '--output-dir',str(tmp_path),'--run-id','disabled'])==0
    assert not (tmp_path/'disabled'/'token_calls.jsonl').exists()
    assert not (tmp_path/'disabled'/'token_summary.json').exists()
