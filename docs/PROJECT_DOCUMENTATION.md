# Project Documentation

## 1. O que este projeto faz

Este projeto implementa um pipeline de verificação com LLM para analisar issues/PRs de projetos open source e identificar discussão substantiva sobre blockchain design patterns.

O pipeline tem dois estágios:

1. Stage 1 (recall): identifica candidatos de patterns por issue/PR.
2. Stage 2 (precision): valida cada par issue-pattern e classifica o veredito final.

Depois disso, o sistema agrega resultados no nível da issue para gerar um dataset pronto para análise quantitativa e revisão humana.

## 2. Como o pipeline funciona

Fluxo de alto nível:

1. Leitura e normalização de entrada CSV.
2. Validação de estrutura (colunas, chave composta e catálogo).
3. Preparação do texto de entrada da issue/PR (incluindo truncamento controlado).
4. Execução do Stage 1.
5. Geração dos pares candidatos para Stage 2.
6. Execução do Stage 2.
7. Agregação final (`issue_results.csv`).
8. Persistência de artefatos de auditoria (manifestos, raw outputs, checkpoints, metadata).

## 3. Estrutura do repositório e papel de cada arquivo

### 3.1 Scripts Python de topo

- [run_pipeline.py](../run_pipeline.py): entrypoint principal do pipeline (subcomandos `validate` e `run`).
- [evaluate_pipeline.py](../evaluate_pipeline.py): compara resultados Stage 1/Stage 2 com anotações humanas e calcula métricas.
- [run_integration_tests.py](../run_integration_tests.py): suíte de integração ponta a ponta (smoke, resumption e data quality).

### 3.2 Arquivos de configuração e dados base

- [requirements.txt](../requirements.txt): dependências Python do projeto.
- [blockchain_patterns_keywords_v3.csv](../blockchain_patterns_keywords_v3.csv): catálogo canônico de patterns usado por Stage 1 e Stage 2.
- [data/pilot/pilot_annotation_sample.csv](../data/pilot/pilot_annotation_sample.csv): amostra piloto principal.
- [data/pilot/pilot_stage1_failed_subset.csv](../data/pilot/pilot_stage1_failed_subset.csv): subconjunto auxiliar para reprocessamento/depuração.
- [data/smoke/smoke_annotation_sample.csv](../data/smoke/smoke_annotation_sample.csv): amostra pequena para testes rápidos.

### 3.3 Pacote principal do pipeline (`llm_pipeline/`)

- [llm_pipeline/__init__.py](../llm_pipeline/__init__.py): inicialização do pacote.
- [llm_pipeline/cli.py](../llm_pipeline/cli.py): parser CLI e orquestração de execução.
- [llm_pipeline/config.py](../llm_pipeline/config.py): constantes, defaults de modelos e taxonomias.
- [llm_pipeline/data.py](../llm_pipeline/data.py): leitura, validação e preparação de dados de entrada.
- [llm_pipeline/normalization.py](../llm_pipeline/normalization.py): normalização robusta de headers/células e compatibilidade CSV.
- [llm_pipeline/prompts.py](../llm_pipeline/prompts.py): prompts dos estágios alinhados ao protocolo metodológico.
- [llm_pipeline/schemas.py](../llm_pipeline/schemas.py): contratos JSON e validações semânticas de resposta.
- [llm_pipeline/requests.py](../llm_pipeline/requests.py): montagem de payloads para a API Gemini.
- [llm_pipeline/client.py](../llm_pipeline/client.py): integração com SDK Gemini, retries e parsing.
- [llm_pipeline/stages.py](../llm_pipeline/stages.py): execução Stage 1/Stage 2 em `sync` e `batch`.
- [llm_pipeline/aggregation.py](../llm_pipeline/aggregation.py): consolidação final por issue (inclui schema final de saída).
- [llm_pipeline/artifacts.py](../llm_pipeline/artifacts.py): geração de snapshots, manifests e metadados.
- [llm_pipeline/models.py](../llm_pipeline/models.py): dataclasses internas do domínio.
- [llm_pipeline/utils.py](../llm_pipeline/utils.py): utilitários gerais (hash, serialização, etc.).
- [llm_pipeline/truncation.py](../llm_pipeline/truncation.py): truncamento head-tail com sinal explícito.
- [llm_pipeline/evidence.py](../llm_pipeline/evidence.py): validação de evidência literal.
- [llm_pipeline/lexical.py](../llm_pipeline/lexical.py): detecção de falsos amigos lexicais.
- [llm_pipeline/confidence.py](../llm_pipeline/confidence.py): override determinístico de confiança.
- [llm_pipeline/checkpoint.py](../llm_pipeline/checkpoint.py): checkpoint e retomada de execução.
- [llm_pipeline/stability.py](../llm_pipeline/stability.py): suporte de rastreabilidade/reprodutibilidade.

### 3.4 Documentação auxiliar

- [README.md](../README.md): guia de setup e execução rápida.
- [docs/ARCHITECTURE.md](ARCHITECTURE.md): visão arquitetural resumida.
- [docs/PIPELINE_STAGE1_STAGE2.md](PIPELINE_STAGE1_STAGE2.md): contrato conceitual de prompts e estágios.
- [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md): guia completo de solução de problemas (quota errors, checkpoint recovery, data issues).
- [INTEGRATION_TESTS_README.md](../INTEGRATION_TESTS_README.md): instruções da suíte de integração.
- [QUICK_START.md](../QUICK_START.md): guia rápido de 5 minutos (clone, setup, primeiro run).
- [DEPLOYMENT_CHECKLIST.md](../DEPLOYMENT_CHECKLIST.md): checklist pré-deployment validação ambiente.
- [REFACTORING_COMPLETE.md](../REFACTORING_COMPLETE.md): histórico de refactoring (material de referência).

### 3.5 Testes

- [tests/test_cli.py](../tests/test_cli.py): testes de parser CLI.
- [tests/test_evaluate.py](../tests/test_evaluate.py): testes da avaliação.
- [tests/test_gemini_integration.py](../tests/test_gemini_integration.py): testes de integração da camada Gemini (mockada).
- [tests/test_gemini_stages.py](../tests/test_gemini_stages.py): testes de execução de estágios.
- [tests/test_normalization.py](../tests/test_normalization.py): testes de normalização.
- [tests/test_pipeline_core.py](../tests/test_pipeline_core.py): testes do core (incluindo agregação).
- [tests/test_refactoring.py](../tests/test_refactoring.py): testes de regressão/refatoração.

## 4. Modos de execução

### 4.1 `validate`

Normaliza e valida o input e o catálogo sem chamar API. Produz relatórios em `outputs/validation/...`.

### 4.2 `run --mode dry-run`

Gera artefatos de preparação e manifests sem inferência real.

### 4.3 `run --mode sync`

Executa chamadas síncronas por item/par. Mais simples de depurar.

### 4.4 `run --mode batch`

Executa via lote (Batch API), adequado para execuções grandes.

## 5. Outputs gerados ao rodar o pipeline

Cada execução cria `outputs/runs/<run_id>/`.

Arquivos principais:

- `input_raw_snapshot.csv`: snapshot do input original.
- `input_clean_snapshot.csv`: input após normalização.
- `pattern_catalog_raw_snapshot.csv`: catálogo original.
- `pattern_catalog_clean_snapshot.csv`: catálogo normalizado.
- `normalization_report.json`: resumo da normalização.
- `normalization_changes.csv`: trilha detalhada de alterações na normalização.
- `pattern_catalog_full.txt`: catálogo completo textual para inspeção.
- `pattern_catalog_compact.txt`: versão compacta do catálogo.
- `request_manifest.json`: manifesto da execução (requests e parâmetros principais).
- `run_metadata.json`: metadados completos (hashes, versões, parâmetros, etc.).
- `stage1_raw.jsonl`: respostas brutas do Stage 1.
- `stage1_results.csv`: resultado estruturado do Stage 1 por issue.
- `stage2_pair_manifest.json`: pares issue-pattern enviados ao Stage 2.
- `stage2_raw.jsonl`: respostas brutas do Stage 2.
- `stage2_results.csv`: resultado estruturado do Stage 2 por par.
- `issue_results.csv`: agregação final por issue (dataset de consumo analítico).
- `run_summary.json`: resumo quantitativo final da run.

Arquivos condicionais:

- `stage1_checkpoint.json` e `stage2_checkpoint.json`: presentes quando há checkpoint/resume.
- `stage1_batches.json` e `stage2_batches.json`: presentes em modo batch.

## 6. Saída final no nível da issue

`issue_results.csv` (ou versões derivadas como `issue_results_schema_v2.csv`) reúne:

- metadados da issue (repositório, datas, labels, body, contagens);
- sinais de pipeline (`issue_summary`, contagens Stage 1/2, erros, truncamento);
- decisão final (`relevant_to_pattern_study`, `patterns_present`, `adoption_status`, etc.);
- campos de evidência e confiança;
- campos reservados para revisão humana (`annotator_id`, `human_notes`, `annotation_date`).

## 7. Execução recomendada em ambiente local

Resumo de comandos:

1. Criar ambiente e instalar dependências.
2. Definir `GEMINI_API_KEY`.
3. Rodar `PYTHONPATH=. .venv/bin/pytest -q`.
4. Rodar `run_pipeline.py validate`.
5. Rodar `run_pipeline.py run` (sync ou batch).
6. (Opcional) Rodar `evaluate_pipeline.py` para comparação com anotações humanas.

## 8. Observações importantes

- O pipeline é auditável: snapshots e raw outputs são preservados.
- O seed melhora repetibilidade, mas não elimina totalmente não determinismo de LLM.
- Em caso de quota, checkpoint permite retomar sem reprocessar tudo.
- `issue_results.csv` depende da qualidade e cobertura de Stage 1/Stage 2.

