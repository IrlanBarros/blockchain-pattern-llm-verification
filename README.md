# Pipeline Gemini de verificação de blockchain design patterns

**Versão:** 0.6.1-gemini  
**Manual metodológico:** 0.2 (2026-07-20)  
**Catálogo:** `blockchain_patterns_keywords_v3.csv` (82 patterns)  
**Provedor:** Gemini Developer API, via SDK oficial `google-genai`

Este projeto executa e audita o pipeline em duas etapas da pesquisa **An Empirical Study of the Adoption and Challenges of Blockchain Design Patterns in OSS Projects**:

1. **Stage 1 — recall:** compreende a issue/PR e gera candidatos plausíveis;
2. **Stage 2 — precisão:** verifica rigorosamente cada par `(issue, pattern)`.

O pacote contém somente a implementação atual para Gemini. Não há arquivos de versões anteriores.

## Modelos padrão

```text
Stage 1: gemini-3.1-flash-lite
Stage 2: gemini-3.5-flash
```

Os dois IDs são versões estáveis e suportam Structured Outputs e Batch API. Os parâmetros padrão são:

```text
temperature: 1.0
seed: 0
Stage 1 thinking level: minimal
Stage 2 thinking level: low
```

A temperatura `1.0` segue a recomendação da família Gemini 3. O `seed` melhora a repetibilidade, mas não garante resultados idênticos em chamadas distintas; por isso o pipeline registra modelos, parâmetros, prompts, schemas e respostas brutas.

## Estrutura

```text
.
├── run_pipeline.py
├── evaluate_pipeline.py
├── blockchain_patterns_keywords_v3.csv
├── requirements.txt
├── llm_pipeline/
│   ├── config.py
│   ├── prompts.py
│   ├── models.py
│   ├── utils.py
│   ├── normalization.py
│   ├── data.py
│   ├── schemas.py
│   ├── requests.py
│   ├── client.py
│   ├── stages.py
│   ├── aggregation.py
│   ├── artifacts.py
│   └── cli.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── PIPELINE_STAGE1_STAGE2.md
└── tests/
```

## 1. Instalação

Recomenda-se Python 3.11 ou 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configuração da chave Gemini

Crie a chave no Google AI Studio e defina apenas uma variável de ambiente:

```bash
read -s GEMINI_API_KEY
export GEMINI_API_KEY
```

Confirme sem imprimir a chave:

```bash
python -c 'import os; print("Configurada" if os.getenv("GEMINI_API_KEY") else "Ausente")'
```

O SDK também aceita `GOOGLE_API_KEY`, mas o projeto interrompe a execução se as duas variáveis existirem com valores diferentes.

Nunca coloque a chave no código, CSV, Git, relatório ou captura de tela.

## 3. CSV do smoke test

Use o CSV original dos 30 registros, não a planilha já anotada. Ele deve conter:

```text
repository
issue_number
issue_title
issue_body
```

A coluna abaixo é opcional, mas deve ser incluída quando disponível:

```text
concatenated_comments
```

Também podem permanecer no arquivo `type`, `labels`, `state`, `html_url`, `sample_group`, `selection_trigger` e outras colunas. Os campos de seleção não entram como rótulos de verdade.

## 4. Testes automatizados

```bash
python -m pytest -q
```

Resultado esperado nesta versão:

```text
28 passed
```

Os testes não acessam a API.

## 5. Validação e normalização

```bash
python run_pipeline.py validate \
  --input data/smoke/smoke_annotation_sample.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --report-dir outputs/validation/smoke
```

O projeto trata, de forma auditável:

- UTF-8, UTF-8 com BOM e CP1252;
- vírgula, ponto e vírgula ou tab como separador;
- espaços, tabs, quebras de linha e espaços Unicode;
- BOM e caracteres invisíveis;
- cabeçalhos com caixa, espaço, hífen ou underscore diferentes;
- células vazias sem transformação indevida em `NaN` textual;
- `issue_number` exportado como `15.0`;
- valores separados por `|`, espaços e duplicatas;
- variantes estruturais de enums e nomes de patterns;
- colisões de colunas e chaves compostas duplicadas.

A normalização não aceita sinônimos e não adivinha semântica. `RBAC`, por exemplo, não é convertido automaticamente em `Role-based control`.

Confirme em `validation_report.json`:

```json
{
  "input_rows": 30,
  "pattern_count": 82,
  "duplicate_keys": 0,
  "status": "ok"
}
```

## 6. Dry-run dos 30 registros

```bash
python run_pipeline.py run \
  --input data/smoke/smoke_annotation_sample.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --mode dry-run \
  --limit 30 \
  --run-id smoke_gemini_dry_run
```

O dry-run não usa a API. Ele preserva snapshots, hashes, parâmetros, schemas, catálogo e as requisições completas do Stage 1.

Revise principalmente:

```text
outputs/runs/smoke_gemini_dry_run/input_clean_snapshot.csv
outputs/runs/smoke_gemini_dry_run/normalization_changes.csv
outputs/runs/smoke_gemini_dry_run/run_metadata.json
outputs/runs/smoke_gemini_dry_run/stage1_requests.jsonl
```

## 7. Primeira chamada real

Teste um registro:

```bash
python run_pipeline.py run \
  --input data/smoke/smoke_annotation_sample.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --mode sync \
  --limit 1 \
  --run-id smoke_gemini_probe_1
```

Depois, teste três registros:

```bash
python run_pipeline.py run \
  --input data/smoke/smoke_annotation_sample.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --mode sync \
  --limit 3 \
  --run-id smoke_gemini_probe_3
```

Não avance se houver `errored`, falha de autenticação, modelo inválido, bloqueio, JSON inválido ou `MAX_TOKENS`.

## 8. Execução real dos 30 registros

```bash
python run_pipeline.py run \
  --input data/smoke/smoke_annotation_sample.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --mode sync \
  --limit 30 \
  --run-id smoke_gemini_30
```

Para o smoke test, use `sync`: o diagnóstico por registro é mais simples. O pipeline fará 30 chamadas no Stage 1 e uma chamada no Stage 2 para cada candidato gerado.

Não use `--overwrite` para substituir uma execução científica. Use outro `--run-id`.

## 9. Modo batch

O modo batch usa requisições inline do Gemini Batch API, com divisão automática por quantidade e tamanho estimado abaixo de 20 MB.

```bash
python run_pipeline.py run \
  --input data/evaluation/development.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --mode batch \
  --run-id pilot_gemini
```

Parâmetros opcionais:

```text
--batch-size 250
--batch-max-bytes 18000000
--poll-seconds 30
```

O Batch API é adequado ao piloto/corpus por ser assíncrono e mais barato. Para 30 casos, prefira `sync`.

## 10. Parâmetros configuráveis

```bash
python run_pipeline.py run --help
```

Exemplo explícito:

```bash
python run_pipeline.py run \
  --input data/smoke/smoke_annotation_sample.csv \
  --mode sync \
  --limit 30 \
  --stage1-model gemini-3.1-flash-lite \
  --stage2-model gemini-3.5-flash \
  --temperature 1.0 \
  --seed 0 \
  --stage1-thinking-level minimal \
  --stage2-thinking-level low \
  --stage1-max-tokens 2048 \
  --stage2-max-tokens 2048 \
  --max-input-chars 12000 \
  --run-id smoke_gemini_30_explicit
```

## 11. Saídas por execução

```text
outputs/runs/<run_id>/
├── input_raw_snapshot.csv
├── input_clean_snapshot.csv
├── pattern_catalog_raw_snapshot.csv
├── pattern_catalog_clean_snapshot.csv
├── normalization_report.json
├── normalization_changes.csv
├── pattern_catalog_full.txt
├── pattern_catalog_compact.txt
├── run_metadata.json
├── request_manifest.json
├── stage1_raw.jsonl
├── stage1_results.csv
├── stage2_pair_manifest.json
├── stage2_raw.jsonl
├── stage2_results.csv
├── issue_results.csv
└── run_summary.json
```

Em batch, também são gerados:

```text
stage1_batches.json
stage2_batches.json
```

Falhas técnicas usam `request_status=errored` e resultam em `pipeline_error` na agregação. Nunca são convertidas em negativos.

## 12. Avaliação contra anotação humana

O conjunto-ouro correto deve ser produzido após A1 e A2 independentes, adjudicação e congelamento. O segundo anotador não deve consultar a saída da LLM antes de congelar sua anotação.

```bash
python evaluate_pipeline.py \
  --human-issues smoke_annotation_adjudicated.csv \
  --human-pairs smoke_pair_annotation_adjudicated.csv \
  --stage1 outputs/runs/smoke_gemini_30/stage1_results.csv \
  --stage2 outputs/runs/smoke_gemini_30/stage2_results.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --output-dir outputs/evaluation/smoke_gemini_30
```

Pares gerados pelo Stage 1 sem rótulo humano são enviados para `pairs_requiring_human_review.csv`; eles não são assumidos como negativos.

## Documentação oficial usada na integração

- SDK Python: https://googleapis.github.io/python-genai/
- Structured Outputs: https://ai.google.dev/gemini-api/docs/structured-output
- Batch API: https://ai.google.dev/gemini-api/docs/batch-api
- Modelos: https://ai.google.dev/gemini-api/docs/models


## Colunas de repositório no smoke test

O CSV do smoke test pode manter as colunas originais do corpus:

- `repository_full_name`: identificador completo do repositório e fonte preferida da chave;
- `repository_category`: metadado preservado para análises posteriores.

Também é aceito o nome legado `repository`. Durante a normalização, o pipeline
cria/sincroniza internamente `repository` e `repository_full_name`. Se ambas
existirem e tiverem valores diferentes, a execução é interrompida para evitar
uma junção incorreta. `repository_category` não é enviada à LLM como evidência
e não participa da chave composta.

As colunas mínimas da entrada são, portanto:

```text
repository_full_name  # ou repository
issue_number
issue_title
issue_body
```
