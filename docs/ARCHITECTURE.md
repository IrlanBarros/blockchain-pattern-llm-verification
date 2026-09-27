# Arquitetura multi-provider do pipeline

## Objetivo da divisão

`run_pipeline.py` apenas encaminha a execução para a CLI. A implementação está dividida por responsabilidade para separar normalização, protocolo metodológico, providers de inferência, execução, agregação e auditoria. O desenho e a operação do backend Llama estão documentados em [LOCAL_LLAMA_DEVELOPMENT.md](LOCAL_LLAMA_DEVELOPMENT.md).

## Fluxo principal

1. `normalization.py` lê o CSV preservando strings, detecta codificação/separador e aplica somente correções estruturais.
2. `data.py` valida colunas, chave composta, catálogo e prepara o texto da issue/PR.
3. `prompts.py` fornece as instruções alinhadas ao Manual v0.2.
4. `schemas.py` constrói JSON Schemas fechados e valida semanticamente as respostas.
5. `requests.py` cria requisições canônicas para os dois stages e as converte para Gemini Batch quando necessário.
6. `providers.py` configura capabilities e adapta OpenAI-compatible HTTP para a resposta canônica; `client.py` seleciona o provider, mantém parsing/retries e o caminho Gemini.
7. `stages.py` executa Stage 1 e Stage 2 em modo síncrono ou batch.
8. `aggregation.py` produz a decisão no nível da issue sem uma terceira chamada de LLM.
9. `artifacts.py` preserva snapshots, hashes, parâmetros, schemas, manifests e metadados.
10. `evaluate_pipeline.py` normaliza arquivos humanos/resultados e calcula métricas apenas em pares rotulados.

## Por que o projeto usa `generateContent`

A execução síncrona e a execução em lote usam a mesma família de requisição do Gemini API. Isso reduz divergências entre o smoke test e o piloto. O Batch API do Gemini opera sobre requisições `generateContent`, por isso essa API é usada nos dois modos.

## Módulos

### `normalization.py`

- normalização Unicode NFKC;
- normalização de line endings;
- remoção de BOM, zero-width characters e controles invisíveis;
- conversão de espaços Unicode em espaço comum;
- compactação de whitespace somente em tokens/categorias;
- normalização conservadora dos cabeçalhos;
- leitura de UTF-8, UTF-8 com BOM e CP1252;
- detecção de vírgula, ponto e vírgula e tab;
- canonicalização por fingerprint estrutural único;
- rejeição de colisões, valores desconhecidos e ambiguidades;
- relatório de cada mudança com preview e hash antes/depois.

### `data.py`

- validação das colunas obrigatórias;
- inclusão de colunas opcionais ausentes;
- preservação de colunas extras do smoke test;
- validação de `(repository, issue_number)` após normalização;
- normalização controlada de `issue_number` no formato `15.0`;
- construção do catálogo completo e compacto;
- truncamento head+tail de corpo e comentários.

### `schemas.py`

- Structured Outputs fechados aos valores do catálogo e das taxonomias;
- descrições de campo para orientar o Gemini;
- canonicalização de variações estruturais;
- validações cruzadas de veredito, adoção, evidência e desafios;
- rejeição de candidato sem evidência/rationale;
- impedimento de atribuição de desafio a par negativo.

### `requests.py`

Cada chamada contém:

- `model`;
- `contents` com o artefato tratado como dado;
- `system_instruction` com regras metodológicas e catálogo;
- `response_mime_type=application/json`;
- `response_json_schema`;
- `temperature`, `seed`, `max_output_tokens` e `thinking_config`.

Para batch, o campo `model` é removido de cada requisição individual e `metadata.custom_id` preserva a associação com a issue/par.

### `client.py`

- cria o cliente a partir de `GEMINI_API_KEY` ou `GOOGLE_API_KEY`;
- rejeita variáveis conflitantes;
- executa retries com backoff;
- extrai JSON, finish reason, response ID, modelo e tokens;
- rejeita respostas bloqueadas, vazias ou terminadas por `MAX_TOKENS`;
- monitora estados do Batch API;
- divide lotes por quantidade e bytes estimados para ficar abaixo do limite inline de 20 MB;
- fecha explicitamente o cliente.

### `stages.py`

- Stage 1: uma chamada por issue/PR;
- Stage 2: uma chamada por candidato;
- preserva respostas brutas;
- erros técnicos geram `request_status=errored`;
- ausência de resposta batch é detectada explicitamente;
- resultados batch são reassociados por `metadata.custom_id`, com fallback posicional apenas quando necessário.

### `artifacts.py`

Cada execução preserva:

- arquivo de entrada original e entrada limpa;
- catálogo original e catálogo limpo;
- relatório e tabela de alterações de normalização;
- hashes dos arquivos, prompts e schemas;
- provedor, família da API, modelos, temperatura, seed e thinking levels;
- versão do SDK;
- manifests, respostas brutas e metadados de batch.

## Parâmetros padrão

```text
provider: google-gemini
api_family: generateContent
stage1_model: gemini-3.1-flash-lite
stage2_model: gemini-3.5-flash
temperature: 1.0
seed: 0
stage1_thinking_level: minimal
stage2_thinking_level: low
```

A temperatura segue a recomendação da família Gemini 3. O seed não garante determinismo absoluto; a reprodutibilidade é tratada como rastreabilidade completa do experimento.

## Limite metodológico

A normalização não muda semântica. Ela não converte `no` em `uncertain`, não escolhe entre `Oracle` e `Reverse Oracle`, não aceita sinônimos e não corrige nomes por similaridade aproximada. Uma transformação só é aplicada quando uma variante estrutural aponta para exatamente um valor canônico. Caso contrário, o pipeline falha explicitamente.


### Compatibilidade com o schema do smoke test

A camada de dados aceita `repository_full_name` como identificador canônico do
corpus e mantém `repository` como alias interno para os stages e joins. A coluna
`repository_category` é preservada nos snapshots, mas não participa da chave
`(repository, issue_number)` e não é usada como evidência pela LLM.

## Perfil experimental de otimização

`--profile optimized` insere `retrieval.py` e `context.py` antes da triagem,
usa `compact_wire.py` para preservar o contrato público com IDs/chaves internos,
e integra `telemetry.py` e `optimization_runtime.py` à execução sync/batch.
`--profile legacy` mantém o comportamento metodológico anterior. As regras,
limitações, artefatos e comandos estão em [TOKEN_OPTIMIZATION.md](TOKEN_OPTIMIZATION.md).
