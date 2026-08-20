# 🧪 Integration Tests

Suíte automática de testes ponta-a-ponta com integração real à API Gemini.

Testa:
- **Smoke Test** (~5-7 min): execução completa do pipeline (Stage 1 + Stage 2)
- **Resumption Test** (~3-5 min): recuperação após interrupção com checkpoint
- **Data Quality** (~1 min): validação de regras de qualidade

---

## ⚡ Quick Start

```bash
cd /home/irlan-barros/faculdade/projetos/llm_verification_pipeline_gemini

# Executar todos os testes (Smoke + Resumption + Quality)
PYTHONPATH=. .venv/bin/python run_integration_tests.py
```

**Tempo estimado**: 10-15 minutos (inclui 2 chamadas completas à API Gemini)  
**Output**: Relátório no console + artefatos em `outputs/runs/`

---

## 📋 O Que Cada Teste Valida

### 1️⃣ Smoke Test
Executa pipeline completo com dados reais (5 registros por padrão):

✅ Stage 1: Geração de candidatos  
✅ Stage 2: Validação de pares  
✅ Agregação: Consolidação por issue  
✅ Checkpointing: JSON persistence  
✅ Output files: Todos os artefatos criados  

**Saída**: Pasta `outputs/runs/integration_smoke_test/`

### 2️⃣ Resumption Test  
Testa capacidade de retomada após interrupção:

✅ Pipeline iniciado  
✅ Interrompido após 120s (durante Stage 1)  
✅ Checkpoint salvo com checkpoint JSON  
✅ Pipeline re-executado (identifica items já processados)  
✅ Skipa Stage 1, continua Stage 2 sem duplicatas  

**Saída**: Pasta `outputs/runs/integration_resumption_test/`

### 3️⃣ Data Quality Validation
Valida que regras de qualidade estão sendo aplicadas:

| Validação | O que testa |
|-----------|-------------|
| Confiança distribuída | Não apenas Confiança "high" |
| False friends | Padrões como "Oracle Database" vs "Oracle" detectados corretamente |
| Schema novo | Colunas `mechanism_match`, `scope_match`, `focus_match` presentes |
| Evidência | Evidence validada e literal |
| PR mapping | Pull requests usam `pull_request_description` |
| Checkpoint | Arquivos `stage1_checkpoint.json` e `stage2_checkpoint.json` criados |

---

## 📊 Interpretar Resultados

### ✅ Sucesso Completo

```
✅ Smoke Test PASSED
✅ Resumption Test PASSED  
✅ Data Quality PASSED

🎉 PRONTO PARA PRODUÇÃO
```

**Próximo passo**: Pode fazer deploy com confiança

### ⚠️ Falha Parcial

```
✅ Smoke Test PASSED
❌ Resumption Test FAILED: Timeout
⚠️ Data Quality PASSED (com warnings)

🔍 Ações necessárias:
1. Aumentar INTEGRATION_RESUMPTION_INTERRUPT_S
2. Revisar logs em outputs/runs/integration_resumption_test/
3. Re-rodar apenas resumption test
```

---

## 🔧 Customizando Testes

### Env vars para controle

```bash
# Limite de registros (default: 5)
INTEGRATION_SMOKE_LIMIT=10 PYTHONPATH=. .venv/bin/python run_integration_tests.py

# Timeout do pipeline em segundos (default: 900)
INTEGRATION_PIPELINE_TIMEOUT_S=1800 PYTHONPATH=. .venv/bin/python run_integration_tests.py

# Tempo antes de interrupção em resumption test (default: 120)
INTEGRATION_RESUMPTION_INTERRUPT_S=180 PYTHONPATH=. .venv/bin/python run_integration_tests.py
```

### Rodar testes individuais

Editar `run_integration_tests.py` e comentar/descomentar testes.

---

## ❌ Troubleshooting

### GEMINI_API_KEY não definido

```bash
export GEMINI_API_KEY="sua-chave-aqui"
PYTHONPATH=. .venv/bin/python run_integration_tests.py
```

### Timeout após 900s

API está lenta ou pipeline teve erro:  
1. Aumentar timeout: `INTEGRATION_PIPELINE_TIMEOUT_S=1800 PYTHONPATH=. .venv/bin/python run_integration_tests.py`  
2. Ver detalhes em `outputs/runs/integration_smoke_test/run_metadata.json`  
3. Ver seção "API Quota Errors" em `docs/TROUBLESHOOTING.md`

### Input file not found

Verificar arquivo existe:
```bash
ls -lh data/pilot/pilot_annotation_sample.csv
```

### Resumption test não consegue interromper

Stage 1 terminou muito rápido:  
```bash
INTEGRATION_SMOKE_LIMIT=20 INTEGRATION_RESUMPTION_INTERRUPT_S=180 PYTHONPATH=. .venv/bin/python run_integration_tests.py
```

---

## � Mais informações

- **README.md**: Guia de setup e execução
- **DEPLOYMENT_CHECKLIST.md**: Checklist pré-deployment
- **docs/TROUBLESHOOTING.md**: Solução de problemas comuns
- **docs/PROJECT_DOCUMENTATION.md**: Documentação técnica completa

### 4. Comparar com Prior Runs
```bash
# Comparar distribuição confiança antes/depois
pandas << 'EOF'
import pandas as pd
old = pd.read_csv("outputs/prior_run/stage2_results.csv")
new = pd.read_csv("outputs/runs/integration_smoke_test/stage2_results.csv")
print("Confidence Distribution Change:")
print(f"Before: {old['confidence'].value_counts().to_dict()}")
print(f"After:  {new['confidence'].value_counts().to_dict()}")
EOF
```

---

## 📝 Arquivo de Log Completo

Se precisar inspecionar detalhes completos dos testes:

```bash
# Ver output do smoke test
tail -100 outputs/runs/integration_smoke_test/*.jsonl

# Ver checkpoint state
cat outputs/runs/integration_smoke_test/stage1_checkpoint.json | jq .

# Ver dados finais
head -20 outputs/runs/integration_smoke_test/stage2_results.csv
```

---

## ⏱️ Tempo de Execução

| Test | Tempo | Notas |
|------|-------|-------|
| Smoke Test | 5-7 min | Depende velocidade API Gemini |
| Resumption Test | 3-5 min | Interrupção @ 90s + resumption |
| Data Quality | <1 min | Análise local dos dados |
| **TOTAL** | **10-15 min** | Incluindo I/O e overhead |

Para rodar **apenas um teste**:

```python
# Editar run_integration_tests.py:
def run_all_tests(self):
    # Comentar os testes que não quer
    # if not self.test_smoke_basic():
    #     all_passed = False
    if not self.test_resumption():  # ← Apenas este
        all_passed = False
```

---

## 📞 Troubleshooting Checklist

- [ ] GEMINI_API_KEY está configurado?
- [ ] Arquivo `data/pilot/pilot_annotation_sample.csv` existe?
- [ ] Espaço em disco suficiente em `outputs/runs/`?
- [ ] Python 3.9+ instalado?
- [ ] Todos os imports funcionam? (`pytest tests/test_refactoring.py`)
- [ ] Conexão internet estável (API Gemini)?

---

## 🎯 Resumo

**Este script automatiza a validação final da refatoração:**

1. ✅ Executa pipeline real com API Gemini
2. ✅ Testa todos os 12 fixes (P1-P12) + 2 features (F1-F2)
3. ✅ Valida checkpointing e resumption
4. ✅ Gera relatório com conclusões claras
5. ✅ Pronto para go/no-go production deployment

**Tempo**: 10-15 min  
**Requisito**: API key válido (quota suficiente para ~35 requisições)

---

**Status**: 🟢 Ready to run  
**Last Updated**: 17/08/2026
