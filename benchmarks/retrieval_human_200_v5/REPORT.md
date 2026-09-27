# Retrieval contra as 200 anotações humanas

Decisão: **APROVAR `lsa_intermediate` como candidata ao A/B real**, não como
garantia populacional nem como autorização para um full run sem supervisão.
Todos os 150 testes locais passam; nenhuma inferência paga foi executada.

O arquivo humano permaneceu somente leitura. Seu SHA-256 é
`fb32632569f2315d5112ee6a2eea78d174cd2d07b221960e19cdf8adb6d5b39f`.
Há 200 issues únicas, sem duplicatas: 7 `yes` (8 pares confirmados), 1
`uncertain` (1 par), 187 `no` e 5 `insufficient_context`. Como as células
preservam decisões `llm:` ao lado de `A1:`, estes dados são descritos como
*human-reviewed pilot annotations*, não como gold standard independente.

| Configuração | Human recall | Misses | Shortlist média | Full fallback | Input estimado (200 + 72) | Redução |
|---|---:|---:|---:|---:|---:|---:|
| Legacy/full catalog | 100% estrutural | 0 | 82,00 | 100% estrutural | 2.047.008 | 0% |
| Conservador atual | 100% (8/8) | 0 | 79,33 | 96,0% | 1.157.461 | 43,46% |
| Agressivo anterior | 100% (8/8) | 0 | 16,11 | 0,5% | 736.604 | 64,02% |
| Neural A (E5) | 87,5% (7/8) | 1 | 31,83 | 0% | 842.205 | 58,86% |
| Neural B (E5 + LSA) | 100% (8/8) | 0 | 29,33 | 4,5% | 825.591 | 59,67% |
| **Recomendado: LSA intermediário** | **100% (8/8)** | **0** | **16,11** | **0,5%** | **736.604** | **64,02%** |

Todas as configurações LSA que aparecem com 100% também recuperaram o par
`uncertain` (9/9 na métrica de segurança). O neural puro perdeu
`Time-Constrained Access` em `OpenZeppelin/openzeppelin-contracts#3735`: score
0,8210, rank 44, sem match lexical nem família configurada. A união com LSA
recupera o caso, mas custa shortlist e tokens; não melhorou o frontier.

O fallback conservador é dominado por `no_lexical_anchor` (190 issues), seguido
por `low_confidence` (60) e `ambiguous_similarity` (1), com sobreposição. Contra
o fallback ampliado, essas regras protegeram zero dos 8 pares humanos conhecidos.
`no_lexical_anchor` adicionou 10.234 entradas de pattern e ~339.242 tokens Stage
1; `low_confidence`, 3.199 e ~106.269; `ambiguous_similarity`, 53 e ~1.777.

O holdout 150/50 não é defensável com apenas 7 issues positivas. Foi usada
validação interna determinística leave-one-positive-issue-out: 8/8 pares
recuperados, zero miss. Isso não é uma estimativa imparcial de desempenho futuro.

O modelo neural opcional é `intfloat/multilingual-e5-small`, revisão
`0e60b8d9d2166d80387f86e3b48ec9ced55f4d15`, licença MIT, dimensão 384. O índice
do catálogo é persistido por fingerprint do modelo/revisão/hash dos pesos,
catálogo, catálogo compacto e versão de normalização. A primeira consulta das
200 issues levou 63,37 s em CPU; LSA intermediário, 4,02–4,58 s. O segundo uso
neural no mesmo processo reutilizou embeddings de consultas em memória e não é
uma medição independente de cold start.

Os tokens são `ceil(caracteres/4)`, não tokens oficiais Gemini. O total usa os
mesmos 72 pares históricos dos benchmarks anteriores apenas como carga fixa de
Stage 2. A retenção histórica do agressivo é 41/72, mas o gold principal é humano.

Artefatos detalhados: `dataset_summary.json`, `human_gold_summary.json`,
`conservative_current.json`, `aggressive_previous.json`, `neural_retrieval.json`,
`parameter_comparison.json`, `retrieval_misses.json`, `fallback_analysis.json`,
`token_comparison.json` e `final_recommendation.json`.
