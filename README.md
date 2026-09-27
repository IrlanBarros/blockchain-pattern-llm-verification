# Pipeline multi-provider de verificação de blockchain design patterns

Pipeline em 2 estágios para identificar e verificar menções a blockchain design patterns em issues/PRs de projetos OSS.

- Stage 1 (recall): lê a issue/PR e propõe candidatos de pattern.
- Stage 2 (precisão): valida cada par `(issue, pattern)` com regras mais estritas.
- Agregação final: consolida no nível da issue (`issue_results.csv`).

## Visão rápida

- Linguagem: Python
- Backends de LLM: Gemini (`google-genai`) e OpenAI-compatible (`llama.cpp`/vLLM)
- Catálogo de patterns: `blockchain_patterns_keywords_v3.csv`
- Entrypoint principal: `run_pipeline.py`

Documentação técnica detalhada do projeto e dos arquivos está em `docs/PROJECT_DOCUMENTATION.md`.

## Local Llama Development

O fluxo OpenAI-compatible para `llama.cpp` CPU-only (e futura migração para
vLLM), incluindo instalação, variáveis, smoke, resume e telemetria, está em
[docs/LOCAL_LLAMA_DEVELOPMENT.md](docs/LOCAL_LLAMA_DEVELOPMENT.md).

## 1. Clonar o projeto

Opção SSH:

```bash
git clone git@github.com:IrlanBarros/blockchain-pattern-llm-verification.git
cd blockchain-pattern-llm-verification
```

Opção HTTPS:

```bash
git clone https://github.com/IrlanBarros/blockchain-pattern-llm-verification.git
cd blockchain-pattern-llm-verification
```

## 2. Preparar ambiente Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configurar o provider

Para desenvolvimento local com Llama, veja o guia completo de arquitetura,
setup CPU-only, smoke, resume e telemetria em
[Local Llama Development](docs/LOCAL_LLAMA_DEVELOPMENT.md).

Para reprodução histórica com Gemini, defina apenas uma variável
(`GEMINI_API_KEY` ou `GOOGLE_API_KEY`).

```bash
export GEMINI_API_KEY="SUA_CHAVE_AQUI"
```

Verificação rápida:

```bash
python -c 'import os; print("OK" if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") else "MISSING")'
```

## 4. Rodar testes

Neste repositório, prefira rodar com `PYTHONPATH=.` para evitar problemas de import.

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

Teste de integração (usa API real):

```bash
PYTHONPATH=. .venv/bin/python run_integration_tests.py
```

## 5. Validar e normalizar um CSV de entrada

```bash
PYTHONPATH=. .venv/bin/python run_pipeline.py validate \
  --input data/smoke/smoke_annotation_sample.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --report-dir outputs/validation/smoke
```

Saídas dessa etapa vão para `outputs/validation/smoke/`.

## 6. Rodar pipeline

### 6.1 Dry-run (sem chamadas de API)

```bash
PYTHONPATH=. .venv/bin/python run_pipeline.py run \
  --input data/smoke/smoke_annotation_sample.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --mode dry-run \
  --run-id smoke_dry_run
```

### 6.2 Execução real (sync)

```bash
PYTHONPATH=. .venv/bin/python run_pipeline.py run \
  --input data/pilot/pilot_annotation_sample.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --mode sync \
  --limit 200 \
  --output-dir outputs/runs/pilot_gemini_user \
  --run-id pilot_sample_run
```

### 6.3 Execução real (batch)

```bash
PYTHONPATH=. .venv/bin/python run_pipeline.py run \
  --input data/pilot/pilot_annotation_sample.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --mode batch \
  --output-dir outputs/runs \
  --run-id pilot_batch_run
```

## 7. Avaliar contra anotações humanas

```bash
PYTHONPATH=. .venv/bin/python evaluate_pipeline.py \
  --human-issues data/human/human_issues_adjudicated.csv \
  --human-pairs data/human/human_pairs_adjudicated.csv \
  --stage1 outputs/runs/pilot_gemini_user/pilot_sample_run/stage1_results.csv \
  --stage2 outputs/runs/pilot_gemini_user/pilot_sample_run/stage2_results.csv \
  --patterns blockchain_patterns_keywords_v3.csv \
  --output-dir outputs/evaluation/smoke
```

## 8. Estrutura de saída por execução

Cada run cria uma pasta `outputs/runs/<run_id>/` com artefatos como:

- `input_raw_snapshot.csv`
- `input_clean_snapshot.csv`
- `normalization_report.json`
- `normalization_changes.csv`
- `request_manifest.json`
- `run_metadata.json`
- `stage1_raw.jsonl`
- `stage1_results.csv`
- `stage2_pair_manifest.json`
- `stage2_raw.jsonl`
- `stage2_results.csv`
- `issue_results.csv`
- `run_summary.json`

## 9. Parâmetros úteis

Ajuda completa:

```bash
PYTHONPATH=. .venv/bin/python run_pipeline.py run --help
```

Flags comuns:

- `--limit`: limita número de registros
- `--run-id`: nome da execução
- `--output-dir`: diretório base dos resultados
- `--overwrite`: permite reutilizar diretório já existente
- `--stage1-model`, `--stage2-model`
- `--temperature`, `--seed`
- `--stage1-thinking-level`, `--stage2-thinking-level`

## 10. Referências rápidas

- Arquitetura: `docs/ARCHITECTURE.md`
- Llama local: `docs/LOCAL_LLAMA_DEVELOPMENT.md`
- Contratos Stage 1/2: `docs/PIPELINE_STAGE1_STAGE2.md`
- Testes de integração: `INTEGRATION_TESTS_README.md`
- Documentação completa do projeto: `docs/PROJECT_DOCUMENTATION.md`

## Otimização experimental de tokens

O perfil opcional `--profile optimized` adiciona retrieval local com fallback,
catálogo compacto, seleção de trechos, IDs internos e telemetria. O perfil
`legacy` continua padrão. A avaliação piloto contra 8 pares humanos confirmados
recuperou 8/8 com a candidata LSA intermediária, shortlist média 16,11, fallback
completo 0,5% e redução estimada de input 64,02%. A amostra positiva é pequena;
a aprovação é apenas para A/B real. Veja [implementação, limites, resultados e
comandos](docs/TOKEN_OPTIMIZATION.md).
