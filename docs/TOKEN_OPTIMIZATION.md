# Otimização de tokens: implementação experimental e validação local

## Atualização: validação humana e embeddings locais

A planilha `data/pilot/human/pilot_annotation_200_human.csv` foi avaliada sem
alteração (SHA-256
`fb32632569f2315d5112ee6a2eea78d174cd2d07b221960e19cdf8adb6d5b39f`). Ela
contém 200 chaves únicas, 8 pares confirmados em 7 issues e um par `uncertain`.
Os campos preservam decisões anteriores da LLM ao lado das decisões `A1`, por
isso a referência é denominada *human-reviewed pilot annotations*, não gold
standard independente.

A candidata `lsa_intermediate` (`RETRIEVAL_CONFIDENCE_THRESHOLD=0.35`,
`RETRIEVAL_REQUIRE_LEXICAL_ANCHOR=false`) recuperou 8/8 confirmados e 9/9 na
métrica de segurança, com shortlist média 16,11, p95 21, fallback completo em
1/200 e redução estimada total de 64,02% sobre a carga fixa de 200 Stage 1 + 72
Stage 2. Ela está **aprovada como candidata ao A/B real**, não como default final.

Foi implementado e medido um backend opcional `semantic_backend=neural` com
`intfloat/multilingual-e5-small` fixado por revisão, cache persistente e chunking.
O neural puro perdeu 1/8 (`Time-Constrained Access`) e teve shortlist 31,83. A
união LSA+neural recuperou 8/8, mas shortlist 29,33 e redução 59,67%; portanto os
embeddings não melhoraram o frontier desta amostra e não são a recomendação.

O diagnóstico completo e reproduzível está em
`benchmarks/retrieval_human_200_v5/REPORT.md` e nos JSONs do mesmo diretório.
A suíte completa contém 150 testes passando. Nenhuma nova inferência Gemini foi
executada nesta etapa.

O restante desta página preserva o diagnóstico e os resultados da rodada anterior
como histórico. As afirmações antigas de que os rótulos estavam vazios e de que
o agressivo não estava aprovado foram superadas pela atualização acima. O perfil
`legacy` continua sendo o default enquanto o A/B real não for executado.

## Diagnóstico do estado inicial

Baseline preservado antes das mudanças em `benchmarks/token_optimization/baseline.json` e `baseline_prompts.json`, com commit e hashes dos dados. A suíte original tinha 103 testes, todos passando.

O fluxo existente era:

1. `normalization.py`: leitura robusta, normalização estrutural e canonicalização sem inferência semântica.
2. `data.py`: validação da chave `(repository, issue_number)`, leitura dos 82 patterns e preparação do artefato.
3. `prepare_issue()`: título único, metadados, corpo e comentários separados; orçamento em caracteres e truncamento head/tail. Não existe `build_text()` nesta versão.
4. `requests.py`: Stage 1 envia o catálogo inteiro, com descrições cortadas em 260 caracteres; Stage 2 envia descrição completa e comparações, repetindo a descrição do candidato.
5. `client.py`: Gemini `generateContent`, JSON estruturado, retries, extração de tokens, execução sync/batch.
6. `stages.py`: validação/normalização das respostas, confiança determinística, detecção lexical informativa, resultados brutos e CSVs.
7. `checkpoint.py`: CSV atômico é a fonte de verdade; checkpoint é reconciliado por identidade composta. Stage 2 usa também o pattern.
8. `aggregation.py`: decisão por issue sem terceira chamada LLM. `evaluate_pipeline.py`: métricas contra rótulos humanos.

O título já aparecia uma única vez. As redundâncias efetivas eram o catálogo completo na triagem, exemplos globais de falsos-amigos, instruções repetidas, nomes longos no JSON e repetição da definição do candidato no Stage 2. Repository/issue number eram enviados como tags, apesar de servirem como metadados de identidade.

O histórico `pilot_sample_run` contém 200 Stage 1 concluídos e 72 candidatos, mas somente 51 resultados Stage 2. Além de incompleto, usa hashes de prompts de uma versão anterior: não é um controle equivalente para afirmar qualidade atual.

## Arquitetura implementada

```text
Dataset normalizado, preservando fontes e chaves
  → corpo, título e comentários originais separados
  → retrieval em TODO o texto (antes de cortar contexto)
      união de nomes/aliases/keywords + TF-IDF/LSA local + famílias
      fallback ampliado ou catálogo completo quando necessário
  → seleção determinística de trechos originais, com offsets
  → Stage 1: catálogo compacto recuperado, falsos-amigos pertinentes
      JSON com chaves curtas e IDs; mantém todos os campos de auditoria
  → decodificação e validação no schema público existente
  → candidatos com nomes canônicos persistidos
  → Stage 2: seleção específica por candidato preservando evidência do Stage 1
      definição completa do candidato + distinções próximas compactas
      todos os vereditos, flags, evidência, justificativa, alternativa e overlap
  → decodificação, validações existentes, CSVs atômicos e agregação
```

O retrieval nunca produz veredito e nunca elimina uma issue. Os limites 5/10 são preferenciais; matches lexicais e famílias podem ultrapassá-los. O fallback padrão envia todos os 82 patterns quando a similaridade é baixa, difusa ou não existe apoio lexical para a recuperação LSA. No piloto isso ocorreu em 192/200 casos. Reduzir esses guardas exige validação humana; não há garantia de recall embutida no algoritmo.

### Catálogo e semântica

- `compact_catalog_v1.json` contém IDs congelados P001–P082 e definições curtas revisáveis, derivadas do catálogo original. Os nomes canônicos sempre vêm do CSV original.
- A definição curta só é usada se o hash da descrição original corresponder. Se o catálogo mudar, usa-se a definição completa, evitando uma síntese obsoleta.
- Novos nomes recebem IDs derivados de hash, sem renumerar os existentes. Colisões são rejeitadas.
- As representações incluem categoria, definição, mecanismos, aliases, keywords, falsos-amigos e famílias; somente o necessário é serializado no prompt.
- LSA projeta TF-IDF em um espaço de coocorrência de até 32 dimensões. É recuperação semântica latente local, não um encoder neural multilíngue. Usa NumPy já presente como dependência do pandas; a dependência direta agora está declarada.
- Nenhum modelo de sumarização foi adicionado. Os excertos são copiados literalmente, com intervalos de origem e marcação explícita de omissão.
- Stage 1 mantém resumo, atividade, desafios, contexto, evidência, localização, rationale e confiança. Retornar somente IDs descartaria auditoria exigida pelo projeto, portanto os IDs compactam o campo pattern sem remover os demais.
- Stage 2 mantém todas as validações metodológicas, inclusive as distinções de mecanismo das regressões existentes. Define o candidato uma única vez. A comparação próxima usa definições compactas; um índice ID/nome permite alternativas em todo o catálogo.
- Prefixos de instruções são estáveis; catálogo e artefato variáveis vêm depois. Não há dependência de cache do provedor nem alegação de desconto automático.

### Resume e auditoria

O perfil otimizado grava `optimization_manifest.json`, `compact_catalog.json` e `request_diagnostics.jsonl`, além dos artefatos atuais. O manifesto registra shortlists, scores, motivos de fallback, hashes da fonte/artefato, offsets selecionados, configuração, versão do NumPy, definições e implementação. A retomada rejeita alteração de implementação, configuração, catálogo, shortlist ou contexto. Não é permitido ativar otimização dentro de uma execução legada existente.

Resultados finais continuam usando nomes canônicos e as mesmas chaves. Registros bem-sucedidos não são reenviados. Erros de modelo ficam recuperáveis; falhas de disco abortam em vez de sobrescrever resultados válidos como erro de inferência.

Jobs batch têm a identidade remota persistida **antes** do polling. Uma retomada pode recuperar o job e processar apenas os resultados pendentes, preservando a ordem original para fallback posicional. Respostas duplicadas não sobrescrevem a primeira resposta processada. Ainda existe a janela inevitável entre aceitar uma criação remota e gravar sua identidade local; a API não oferece aqui uma transação atômica conjunta.

### Telemetria

`token_calls.jsonl` registra cada tentativa sync e cada resposta batch: stage, repository, issue number, pattern, modelo, uso oficial quando disponível, pensamento/cache, quantidade de candidatos, caracteres do prompt incluindo schema, estimativa, latência e erros de transporte. Respostas inválidas também podem consumir tokens e entram na contabilidade antes do parsing semântico.

`token_summary.json` agrega input/output/total, pensamento/cache, média/mediana/p95 por stage, candidatos por issue e completude de uso. Custos só são estimados se tarifas explícitas forem fornecidas. Uso ausente é desconhecido, nunca zero. `execution_sessions.jsonl` registra tempo de parede por invocação, incluindo retomadas. Latência individual em batch permanece nula, pois não é observável por resposta.

O estimador offline e os orçamentos de contexto usam `ceil(caracteres/4)`. **Não são o tokenizer oficial Gemini nem limites exatos de tokens.** Não estão instalados sentencepiece ou um tokenizer local compatível com os modelos configurados. A contagem autoritativa durante execução vem de `usage_metadata` do provedor. As métricas offline não autorizam afirmar economia real de faturamento, output ou tokens de pensamento.

## Resultados medidos

A comparação local principal usa as mesmas 200 issues e os mesmos **72 pares históricos** para construir prompts Stage 2 nos dois perfis. Não reduzimos artificialmente a carga Stage 2 removendo pares excluídos pelo retrieval.

| Métrica | Baseline atual | Otimizado conservador | Redução |
|---|---:|---:|---:|
| Input estimado médio Stage 1 | 8.409,59 | 4.460,59 | 46,96% |
| Input estimado médio Stage 2 | 5.097,15 | 3.700,96 | 27,39% |
| Input estimado total: 200 + 72 chamadas | 2.048.912 | 1.158.587 | **43,45%** |
| Mediana input Stage 1 | 8.218,50 | 4.351,50 | — |
| P95 input Stage 1 | 9.320,90 | 5.563,10 | — |
| Patterns apresentados por issue | 82 | 79,325 | — |
| Candidatos históricos excluídos | — | **0 de 72** | Não é recall humano |
| Tempo local de preparação de prompts | 0,119 s | 5,010 s | Aumentou; não mede inferência |
| Output/total real e tempo de inferência | Não executado | Não executado | Não medido |
| Candidate recall Stage 1 | Não medido | Não medido | — |
| Precision / recall / F1 Stage 2 | Não medidos | Não medidos | — |

No smoke, a redução estimada média é 46,85% no Stage 1 e 27,71% no Stage 2. O probe de Stage 2 cobre 30×82=2.460 pares: seu total é um teste de construção de prompts, **não** o custo de um fluxo real com candidatos produzidos pelo Stage 1.

Os tempos locais de construção, intervalos completos, metadados e scores estão nos relatórios JSON. Não medem latência de inferência e variam com carga/CPU. O retrieval adiciona trabalho local: não se promete redução de tempo total sem medir o modelo.

Para referência exclusivamente histórica, o run anterior reportou pela API médias de 7.048,09 input / 124,67 output no Stage 1 e 2.563,59 input / 209,27 output no Stage 2. O total disponível foi 1.575.966 tokens, cobrindo 200 + 51 chamadas concluídas. Esses números **não** entram como denominador do benchmark atual, por diferenças de prompts e cobertura.

### Experimento agressivo — decisão histórica anterior aos rótulos humanos

Com `RETRIEVAL_CONFIDENCE_THRESHOLD=0.18` e `RETRIEVAL_REQUIRE_LEXICAL_ANCHOR=false`, o input estimado passou de 2.048.912 para 737.963 (**63,98%**), com shortlist média de 16,14 patterns. Porém, 31/72 candidatos históricos foram excluídos. Todos estão listados individualmente em `pilot_aggressive_not_approved/report.json`, no campo `historical_llm_candidates_excluded`.

Esses 31 casos não são falsos negativos humanos demonstrados: faltam rótulos. São divergências que precisam de inspeção antes de qualquer adoção. O modo conservador foi escolhido em vez de apresentar esse ganho como uma otimização aprovada.

### Aceitação e riscos da rodada anterior (superados em parte pela atualização)

- **Eficiência >=60%:** não atendida na configuração conservadora; economia real total não medida.
- **Retrieval recall >=98%:** desconhecido; zero positivos humanos rotulados disponíveis. Zero de 72 candidatos antigos excluídos não substitui essa métrica.
- **Stage 1 queda <=1 pp / Stage 2 <=1–2 pp:** desconhecidas sem inferência pareada e rótulos.
- **Testes:** 141 passaram; incluem integração simulada, parsing, IDs, seleção de comentários, falhas parciais, telemetria, configuração e resume sync/batch. Não demonstram qualidade de um modelo remoto.
- LSA pode perder paráfrases e relações fora do vocabulário do catálogo. Fallback é uma mitigação, não uma garantia matemática de recall.
- Seleção extrativa pode omitir contexto necessário, inclusive negações fora do trecho selecionado. Há marcação de omissão e proteção das evidências do Stage 1, mas é necessária revisão de casos.
- Definições curtas e instruções compactas podem alterar decisões da LLM, mesmo quando todos os campos e critérios são preservados.
- As famílias existentes cobrem sobreposições conhecidas; não constituem ontologia completa.
- O estimador por caracteres pode errar especialmente em código, Unicode e mudanças de idioma. Não usar suas porcentagens como evidência de redução faturada.

Não foram alterados dados brutos, decisões humanas, resultados anteriores ou métricas congeladas.

## Arquivos

| Arquivo criado/modificado | Responsabilidade |
|---|---|
| `llm_pipeline/optimization.py` | Perfil explícito e configuração validada por ambiente, com fingerprint |
| `llm_pipeline/compact_catalog_v1.json` | IDs estáveis, definições curtas, hashes da fonte |
| `llm_pipeline/retrieval.py` | Catálogo compacto, lexical, LSA, união, famílias e fallback |
| `llm_pipeline/context.py` | Ranking extrativo de corpo/comentários com offsets e proteção de evidência |
| `llm_pipeline/compact_wire.py` | Chaves internas compactas e conversão bidirecional ID → canônico |
| `llm_pipeline/optimization_runtime.py` | Manifestos, hashes, guardas de resume e integração dos stages |
| `llm_pipeline/telemetry.py` | Contabilidade por tentativa/resposta e sumários |
| `llm_pipeline/models.py` | Contexto-fonte e diagnósticos internos opcionais |
| `llm_pipeline/prompts.py` | Prompts otimizados; textos legados preservados |
| `llm_pipeline/requests.py` | Shortlist e falsos-amigos seletivos; prefixo estável; schema interno |
| `llm_pipeline/stages.py` | Integração, parsing, extração Stage 2 e telemetria sync/batch |
| `llm_pipeline/client.py` | Observação de cada tentativa, incluindo retries |
| `llm_pipeline/checkpoint.py` | Persistência e retomada de jobs remotos batch |
| `llm_pipeline/cli.py` | Perfis, execução separada dos stages, retomada e status de integridade |
| `llm_pipeline/artifacts.py` | Proveniência, commit, quantização e dry-run idempotente |
| `benchmark_tokens.py` | Benchmark offline pareado, distribuição e diagnóstico de retrieval |
| `compare_pipeline_runs.py` | Comparação de runs reais com rótulos e critérios de aceitação explícitos |
| `tests/test_token_optimization.py` | 38 testes novos, além dos 103 existentes |
| `requirements.txt` | Declara NumPy como dependência direta já utilizada via pandas |
| `benchmarks/token_optimization/` | Baseline anterior às alterações e três relatórios reproduzíveis |
| `docs/TOKEN_OPTIMIZATION.md`, `README.md`, `docs/ARCHITECTURE.md` | Documentação, limites e comandos |

Nenhum arquivo existente foi removido.

## Configuração

Ativação: `--profile optimized`. `--profile legacy` preserva prompts e preparação originais. Os campos abaixo são lidos uma vez por execução, validados e persistidos; mudar parâmetros de um run otimizado exige um novo `--run-id`.

```text
RETRIEVAL_ENABLED=true
RETRIEVAL_MIN_CANDIDATES=5
RETRIEVAL_MAX_CANDIDATES=10
RETRIEVAL_SIMILARITY_THRESHOLD=0.18
RETRIEVAL_CONFIDENCE_THRESHOLD=0.55
RETRIEVAL_REQUIRE_LEXICAL_ANCHOR=true
RETRIEVAL_AMBIGUITY_RATIO=0.8
RETRIEVAL_FALLBACK_MULTIPLIER=2
RETRIEVAL_FAMILY_EXPANSION=true
FULL_CATALOG_FALLBACK=true
FORCE_FULL_CATALOG=false
SEMANTIC_DIMENSIONS=32
SEMANTIC_BACKEND=lsa
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_MODEL_REVISION=0e60b8d9d2166d80387f86e3b48ec9ced55f4d15
EMBEDDING_CACHE_DIR=.cache/retrieval_embeddings
EMBEDDING_LOCAL_FILES_ONLY=true
EMBEDDING_BATCH_SIZE=16
SEMANTIC_CHUNK_CHARS=1200
STAGE1_MAX_CONTEXT_TOKENS=1600
STAGE2_MAX_CONTEXT_TOKENS=2400
COMMENT_RETRIEVAL_ENABLED=true
CONTEXT_CHUNK_CHARS=800
BODY_BUDGET_FRACTION=0.75
COMPACT_OUTPUT=true
TOKEN_TELEMETRY_ENABLED=true
```

`FULL_CATALOG_FALLBACK` permite fallback de baixa confiança; `FORCE_FULL_CATALOG` o aplica a todos os casos. Desabilitar retrieval ainda usa catálogo compacto completo, preparação otimizada e os demais controles. `INPUT_PRICE_PER_MILLION` e `OUTPUT_PRICE_PER_MILLION` são opcionais; não há tabela de preços presumida. O cálculo usa tarifas planas fornecidas pelo operador e não incorpora descontos de cache/batch.

## Comandos reproduzíveis

Execute na raiz do projeto. Os benchmarks recusam sobrescrever uma pasta existente: escolha outra pasta para uma nova medição. Não altere variáveis de configuração entre etapas/retomadas do mesmo run.

Testes:

```bash
.venv/bin/python -m pytest -q
```

Avaliação humana completa (offline). Para incluir o backend neural, instale as
dependências opcionais e faça o download/cache inicial do modelo; depois use um
diretório novo a cada rodada:

```bash
.venv/bin/pip install -r requirements-neural.txt

PYTHONPATH=. .venv/bin/python benchmark_retrieval_human.py \
  --human data/pilot/human/pilot_annotation_200_human.csv \
  --historical-stage1 outputs/runs/pilot_gemini_user/pilot_sample_run/stage1_results.csv \
  --output-dir benchmarks/retrieval_human_200_reproduction
```

Sem a dependência/modelo neural, o benchmark LSA continua disponível:

```bash
PYTHONPATH=. .venv/bin/python benchmark_retrieval_human.py \
  --human data/pilot/human/pilot_annotation_200_human.csv \
  --historical-stage1 outputs/runs/pilot_gemini_user/pilot_sample_run/stage1_results.csv \
  --output-dir benchmarks/retrieval_human_200_lsa_only \
  --skip-neural
```

Benchmark local, sem credenciais e sem inferência:

```bash
.venv/bin/python benchmark_tokens.py \
  --input data/pilot/pilot_annotation_sample.csv \
  --pairs-from-stage1 outputs/runs/pilot_gemini_user/pilot_sample_run/stage1_results.csv \
  --output-dir outputs/benchmarks/pilot_local_new

.venv/bin/python benchmark_tokens.py \
  --input data/smoke/smoke_annotation_sample.csv \
  --output-dir outputs/benchmarks/smoke_local_new
```

O primeiro comando requer o CSV histórico e seu `run_metadata.json` já presentes neste workspace. Em outro checkout sem esses artefatos, omita `--pairs-from-stage1`: será explicitamente um probe de todos os pares, não uma estimativa do fluxo real.

Inspeção de requests sem chamar API:

```bash
.venv/bin/python run_pipeline.py run \
  --input data/smoke/smoke_annotation_sample.csv \
  --profile optimized --mode dry-run --run-id smoke_opt_inspect
```

Stage 1 isolado e, depois, Stage 2 a partir dos candidatos persistidos:

```bash
.venv/bin/python run_pipeline.py run \
  --input data/pilot/pilot_annotation_sample.csv \
  --profile optimized --mode sync --stage stage1 --run-id pilot_opt_v1

.venv/bin/python run_pipeline.py run \
  --input data/pilot/pilot_annotation_sample.csv \
  --profile optimized --mode sync --stage stage2 --run-id pilot_opt_v1
```

Pipeline completo; execute exatamente o mesmo comando para retomar uma interrupção:

```bash
.venv/bin/python run_pipeline.py run \
  --input data/pilot/pilot_annotation_sample.csv \
  --profile optimized --mode sync --run-id pilot_opt_full_v1
```

Para batch, use `--mode batch` desde a criação do run; o processamento e o resume usam os mesmos contratos de dados. As execuções reais precisam de `GEMINI_API_KEY` ou `GOOGLE_API_KEY` no ambiente, sem expor a chave em arquivos ou no chat.

A/B real, quando houver credenciais e rótulos preenchidos:

```bash
.venv/bin/python run_pipeline.py run \
  --input data/pilot/pilot_annotation_sample.csv \
  --profile legacy --mode sync --run-id pilot_ab_legacy_v1

RETRIEVAL_CONFIDENCE_THRESHOLD=0.35 \
RETRIEVAL_REQUIRE_LEXICAL_ANCHOR=false \
SEMANTIC_BACKEND=lsa \
.venv/bin/python run_pipeline.py run \
  --input data/pilot/pilot_annotation_sample.csv \
  --profile optimized --mode sync --run-id pilot_ab_opt_v1

.venv/bin/python compare_pipeline_runs.py \
  --baseline-run outputs/runs/pilot_ab_legacy_v1 \
  --optimized-run outputs/runs/pilot_ab_opt_v1 \
  --human-issues data/pilot/human/pilot_annotation_200_human.csv \
  --output outputs/benchmarks/pilot_live_comparison_v1.json
```

`--human-pairs caminho/pares_revisados.csv` pode complementar ou substituir o arquivo de issues. Ele deve conter `repository` (ou alias aceito), `issue_number`, `pattern`, `human_verdict`. Sem pares negativos rotulados, a precisão tem cobertura limitada e não representa todos os candidatos gerados.

O comparador verifica compatibilidade de amostra, catálogo, modelos, temperatura, seed, parâmetros de geração e modo; mede tokens reais dos ledgers completos; lista novos misses; reporta métricas condicionais Stage 2 e métricas end-to-end separadas. No cálculo end-to-end estrito, somente `yes` é positivo; candidatos filtrados e abstenções contam como não-yes, explicitamente. Não altera o avaliador existente. Código de saída 2 significa critérios não aprovados ou não medidos. Aprovação em uma amostra nunca implica garantia populacional de recall.
