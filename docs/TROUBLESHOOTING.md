# Troubleshooting — Guia de Soluções

Guia prático para resolver problemas comuns executando o pipeline.

---

## 🔴 API & Quota Errors

### ❌ RESOURCE_EXHAUSTED: Quota excedida

**Mensagem típica:**
```
Error 429: "RESOURCE_EXHAUSTED: The resource has been exhausted."
```

**Causa**: Atingiu limite de requisições à API Gemini (quota).

**Solução**:

1. **Retomar após esperar**: O pipeline tem checkpoint automático
   ```bash
   # Esperar 30-60 minutos para quota resetar
   sleep 3600
   
   # Re-executar comando original (detectará checkpoint e retomará)
   PYTHONPATH=. .venv/bin/python run_pipeline.py run \
     --input data/pilot/pilot_annotation_sample.csv \
     --patterns blockchain_patterns_keywords_v3.csv \
     --mode sync \
     --output-dir outputs/runs/pilot_gemini_user \
     --run-id pilot_sample_run
   ```

2. **Verificar checkpoint criado**:
   ```bash
   ls -lh outputs/runs/pilot_gemini_user/pilot_sample_run/stage*_checkpoint.json
   ```
   
   Se arquivo existe: checkpoint pronto. Re-execute acima.

3. **Usar quota monitoring**:
   ```bash
   # Ver quantas requisições foram feitas
   cat outputs/runs/pilot_gemini_user/pilot_sample_run/run_metadata.json | grep -i "stage1_count\|stage2_count"
   ```

4. **Limitar próxima tentativa**:
   ```bash
   # Rodar com --limit 10 para testar com fewer requests
   PYTHONPATH=. .venv/bin/python run_pipeline.py run \
     --input data/pilot/pilot_annotation_sample.csv \
     --patterns blockchain_patterns_keywords_v3.csv \
     --mode sync \
     --limit 10 \
     --output-dir outputs/runs/pilot_test_limited \
     --run-id test_limited
   ```

---

### ❌ UNAUTHENTICATED: API Key inválida

**Mensagem típica:**
```
Error 401: "UNAUTHENTICATED"
```

**Causa**: `GEMINI_API_KEY` ou `GOOGLE_API_KEY` não configurado ou inválido.

**Solução**:

1. **Verificar se key existe**:
   ```bash
   echo $GEMINI_API_KEY
   ```
   
   Se vazio: configure:
   ```bash
   export GEMINI_API_KEY="sua-chave-aqui"
   ```

2. **Verificar formato**:
   ```bash
   # Key deve ter ~35+ caracteres
   echo $GEMINI_API_KEY | wc -c
   ```
   
   Se menos de 20 caracteres: key inválida, obter nova.

3. **Testar com simple request**:
   ```bash
   PYTHONPATH=. .venv/bin/python -c "
from llm_pipeline.client import GeminiClient
import os
key = os.getenv('GEMINI_API_KEY')
client = GeminiClient(api_key=key)
print('✓ API key OK' if key else '✗ Missing key')
   "
   ```

4. **Obter nova chave**:
   - Ir para https://ai.google.dev/
   - Fazer login
   - Criar nova API key em "API Keys"
   - Substituir em environment

---

## 🔴 Checkpoint & Resumption

### ❌ Checkpoint não detectado

**Sintoma**: Ao re-executar após falha, pipeline começa do zero (não retoma).

**Causa**: Checkpoint JSON não foi salvo ou está corrompido.

**Solução**:

1. **Verificar se checkpoint existe**:
   ```bash
   ls -lh outputs/runs/<run_id>/stage*_checkpoint.json
   ```
   
   Se não existe: pipeline não criou checkpoint. Ver seção "Checkpoint não criado" abaixo.

2. **Verificar integridade do JSON**:
   ```bash
   python -m json.tool outputs/runs/<run_id>/stage1_checkpoint.json > /dev/null
   ```
   
   Se erro: JSON corrompido. Deletar e re-executar:
   ```bash
   rm outputs/runs/<run_id>/stage*_checkpoint.json
   # Re-executar run com --overwrite
   ```

3. **Forçar retomada**:
   ```bash
   # Usar --overwrite para permitir reutilizar diretório
   PYTHONPATH=. .venv/bin/python run_pipeline.py run \
     --input data/pilot/pilot_annotation_sample.csv \
     --patterns blockchain_patterns_keywords_v3.csv \
     --mode sync \
     --output-dir outputs/runs/pilot_gemini_user \
     --run-id pilot_sample_run \
     --overwrite
   ```

---

### ❌ Checkpoint não criado

**Sintoma**: Arquivo `stage1_checkpoint.json` ou `stage2_checkpoint.json` não aparece.

**Causa**: Pipeline interrompido muito cedo (antes de completar primeiro stage) ou erro durante checkpoint.

**Solução**:

1. **Verificar logs de erro**:
   ```bash
   tail -100 outputs/runs/<run_id>/run_metadata.json
   ```
   
   Ver se há mensagens de erro após "stage1" ou "stage2".

2. **Re-executar com modo verbose**:
   ```bash
   PYTHONPATH=. .venv/bin/python run_pipeline.py run \
     --input data/pilot/pilot_annotation_sample.csv \
     --patterns blockchain_patterns_keywords_v3.csv \
     --mode sync \
     --limit 5 \
     --output-dir outputs/runs/debug_checkpoint
   ```
   
   Procurar por "Checkpoint" nas saídas.

3. **Verificar espaço em disco**:
   ```bash
   df -h outputs/
   ```
   
   Se menos de 1GB livre: liberar espaço (deletar runs antigos).

---

## 🔴 Input & Data Errors

### ❌ FileNotFoundError: Input CSV não encontrado

**Mensagem típica:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/pilot/xxx.csv'
```

**Solução**:

1. **Verificar caminho**:
   ```bash
   ls -lh data/pilot/pilot_annotation_sample.csv
   ```

2. **Usar caminho absoluto se necessário**:
   ```bash
   PYTHONPATH=. .venv/bin/python run_pipeline.py run \
     --input $(pwd)/data/pilot/pilot_annotation_sample.csv \
     --patterns $(pwd)/blockchain_patterns_keywords_v3.csv \
     --mode dry-run
   ```

3. **Verificar repo clonado corretamente**:
   ```bash
   git status  # Deve estar limpo
   du -sh data/  # Deve ter conteúdo
   ```

---

### ❌ Colunas faltando no CSV

**Mensagem típica**:
```
Error: Required column 'project_name' not found
```

**Solução**:

1. **Validar CSV antes de executar**:
   ```bash
   PYTHONPATH=. .venv/bin/python run_pipeline.py validate \
     --input data/pilot/pilot_annotation_sample.csv \
     --patterns blockchain_patterns_keywords_v3.csv \
     --report-dir outputs/validation/debug
   ```
   
   Ver detalhes em `outputs/validation/debug/normalization_report.json`

2. **Verificar estrutura esperada**:
   ```bash
   head -3 data/pilot/pilot_annotation_sample.csv | cut -d',' -f1-10
   ```
   
   Deve ter: `project_name,issue_number,artifact_type,title,body`

3. **Confirmar encoding**:
   ```bash
   file -i data/pilot/pilot_annotation_sample.csv
   ```
   
   Deve ser `text/plain; charset=utf-8`

---

### ❌ CSV com encoding incorreto (UTF-8 BOM, CP1252)

**Sintoma**: Caracteres estranhos na primeira linha ou erro "encoding".

**Solução**:

1. **Converter para UTF-8 limpo**:
   ```bash
   # macOS/Linux
   iconv -f CP1252 -t UTF-8 data/pilot/input.csv > data/pilot/input_clean.csv
   mv data/pilot/input_clean.csv data/pilot/input.csv
   
   # Ou remover BOM
   sed -i '1s/^\xEF\xBB\xBF//' data/pilot/input.csv
   ```

2. **Re-validar**:
   ```bash
   PYTHONPATH=. .venv/bin/python run_pipeline.py validate \
     --input data/pilot/pilot_annotation_sample.csv \
     --patterns blockchain_patterns_keywords_v3.csv
   ```

---

## 🔴 Execution Errors

### ❌ PYTHONPATH error: Cannot find module

**Mensagem típica**:
```
ModuleNotFoundError: No module named 'llm_pipeline'
```

**Solução**:

1. **Sempre use `PYTHONPATH=.`**:
   ```bash
   # ❌ Errado
   python run_pipeline.py ...
   
   # ✅ Correto
   PYTHONPATH=. .venv/bin/python run_pipeline.py ...
   ```

2. **Verificar venv ativado**:
   ```bash
   which python
   # Deve apontar para .venv/bin/python
   ```

3. **Recriar venv se necessário**:
   ```bash
   rm -rf .venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

### ❌ KeyError durante normalização

**Mensagem típica**:
```
KeyError: 'body'
```

**Solução**:

1. **Rodar validação completa**:
   ```bash
   PYTHONPATH=. .venv/bin/python run_pipeline.py validate \
     --input data/pilot/pilot_annotation_sample.csv \
     --patterns blockchain_patterns_keywords_v3.csv \
     --report-dir outputs/validation/full
   ```

2. **Revisar report**:
   ```bash
   cat outputs/validation/full/normalization_report.json | python -m json.tool
   ```

3. **Verificar registros problemáticos**:
   ```bash
   # Ver CSV com linha e coluna
   cat -n data/pilot/pilot_annotation_sample.csv | head -20
   ```

---

### ❌ Memory Error (OOM)

**Mensagem típica**:
```
MemoryError: Unable to allocate XXX GiB for an array
```

**Solução**:

1. **Limitar número de registros**:
   ```bash
   PYTHONPATH=. .venv/bin/python run_pipeline.py run \
     --input data/pilot/pilot_annotation_sample.csv \
     --patterns blockchain_patterns_keywords_v3.csv \
     --limit 50 \
     --mode sync
   ```

2. **Executar em chunks**:
   ```bash
   # Rodar 100 registros de cada vez
   for limit in 1-100 101-200 201-300; do
     PYTHONPATH=. .venv/bin/python run_pipeline.py run \
       --input data/pilot/pilot_annotation_sample.csv \
       --patterns blockchain_patterns_keywords_v3.csv \
       --limit 100 \
       --output-dir outputs/runs/chunk_$limit
   done
   ```

3. **Verificar sistema**:
   ```bash
   free -h  # Memória disponível
   ps aux --sort=-%mem | head  # Outros processos usando RAM
   ```

---

## 🔴 CLI & Configuration

### ❌ Invalid argument error

**Mensagem típica**:
```
error: unrecognized arguments: --old-flag
```

**Solução**:

1. **Ver ajuda completa**:
   ```bash
   PYTHONPATH=. .venv/bin/python run_pipeline.py run --help
   ```

2. **Verificar que subcomando não é confundido**:
   ```bash
   # Subcomandos válidos: validate, run
   PYTHONPATH=. .venv/bin/python run_pipeline.py validate --help
   PYTHONPATH=. .venv/bin/python run_pipeline.py run --help
   ```

3. **Usar argumentos corretos**:
   ```bash
   # --mode pode ser: dry-run, sync, batch
   # --run-id é string (sem espaços)
   # --limit é número inteiro
   ```

---

### ❌ --human-pairs missing (evaluate_pipeline.py)

**Mensagem típica**:
```
error: the following arguments are required: --human-pairs
```

**Solução**:

1. **Sempre informar ambos: `--human-issues` e `--human-pairs`**:
   ```bash
   PYTHONPATH=. .venv/bin/python evaluate_pipeline.py \
     --human-issues data/human/human_issues_adjudicated.csv \
     --human-pairs data/human/human_pairs_adjudicated.csv \
     --stage1 outputs/runs/X/stage1_results.csv \
     --stage2 outputs/runs/X/stage2_results.csv \
     --patterns blockchain_patterns_keywords_v3.csv \
     --output-dir outputs/evaluation
   ```

---

## 🔴 Output & Testing

### ❌ Arquivos de saída vazios ou incompletos

**Sintoma**: `stage1_results.csv` tem 0 linhas ou faltam colunas.

**Solução**:

1. **Verificar run_metadata.json**:
   ```bash
   cat outputs/runs/<run_id>/run_metadata.json | python -m json.tool
   ```
   
   Ver `stage1_count` e verificar se > 0.

2. **Modo dry-run nunca gera resultados reais**:
   ```bash
   # ❌ Errado: dry-run não chama API
   PYTHONPATH=. .venv/bin/python run_pipeline.py run --mode dry-run
   
   # ✅ Correto: sync chama API
   PYTHONPATH=. .venv/bin/python run_pipeline.py run --mode sync
   ```

3. **Verificar se há erros no raw**:
   ```bash
   head -5 outputs/runs/<run_id>/stage1_raw.jsonl | python -m json.tool
   ```

---

### ❌ Testes falhando

**Solução**:

1. **Rodar testes com prints**:
   ```bash
   PYTHONPATH=. .venv/bin/pytest tests/ -v -s
   ```

2. **Rodar teste específico**:
   ```bash
   PYTHONPATH=. .venv/bin/pytest tests/test_pipeline_core.py::test_normalization -v
   ```

3. **Ver arquivo de teste para entender**:
   ```bash
   cat tests/test_pipeline_core.py | head -50
   ```

---

## 🟢 Debug & Monitoring

### 📊 Ver progresso durante execução

```bash
# Terminal 1: Executar pipeline
PYTHONPATH=. .venv/bin/python run_pipeline.py run --input ... --mode sync

# Terminal 2: Monitorar output
watch -n 1 'ls -lh outputs/runs/*/stage*_raw.jsonl'

# Terminal 3: Ver linhas geradas
watch -n 2 'wc -l outputs/runs/*/*_results.csv'
```

---

### 📊 Verificar consumo de API

```bash
# Ver quantas requisições foram feitas
cat outputs/runs/<run_id>/request_manifest.json | python -m json.tool | grep -i "request_count\|total"

# Ver tempo decorrido
python -c "
import json
with open('outputs/runs/<run_id>/run_metadata.json') as f:
    meta = json.load(f)
    start = meta.get('timestamp_start', 'N/A')
    end = meta.get('timestamp_end', 'N/A')
    print(f'Start: {start}, End: {end}')
"
```

---

### 🔍 Verificar distribuição de confiança

```bash
# Ver confiança distribution
PYTHONPATH=. .venv/bin/python -c "
import pandas as pd
df = pd.read_csv('outputs/runs/<run_id>/stage2_results.csv')
print(df['confidence'].value_counts().sort_index(ascending=False))
"
```

---

## 📞 Ainda Precisa Ajuda?

1. **Ver logs completos**: `outputs/runs/<run_id>/run_metadata.json`
2. **Ler docs**: `docs/PROJECT_DOCUMENTATION.md`, `README.md`
3. **Investigar código**: módulos em `llm_pipeline/` têm docstrings
4. **Reporte issue**: Incluir `run_metadata.json` + erro exato + comando executado
