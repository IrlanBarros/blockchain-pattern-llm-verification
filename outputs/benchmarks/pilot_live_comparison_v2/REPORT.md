# Auditoria issue-level — pilot A/B v2

> Referência: **human-reviewed pilot annotations**; não é gold standard independente.

## Síntese objetiva

- Issues: 200 (yes=7, no=187, uncertain=1, insufficient_context=5); avaliação binária n=194.
- Positivas finais: legacy=15; optimized=18. O número 18 é final: 7 TP + 10 FP + 1 positiva humana uncertain excluída da matriz.
- Legacy: TP=5, FP=9, FN=2, TN=178; precision=35.71%, recall=71.43%, F1=47.62%, specificity=95.19%, FPR=4.81%.
- Optimized: TP=7, FP=10, FN=0, TN=177; precision=41.18%, recall=100.00%, F1=58.33%, specificity=94.65%, FPR=5.35%.
- Divergências binárias únicas: 16. Patterns FP optimized: Reverse Oracle (2), State initialization (2), Factory contract (1), Identifier Registry (1), Proxy contract (1), Role-based control (1), Self-Confirmed Transactions (1), Zero-knowledge proof (1).
- Diagnósticos FP optimized: {'insufficient_pattern_discussion': 2, 'background_architecture': 3, 'human_model_disagreement': 2, 'false_friend': 3}. Origem operacional: {'stage2_false_positive': 10}.

A agregação reconstituída confirma que toda e somente issue com pelo menos um Stage 2 `yes` recebeu decisão final `yes`. Não foi encontrado bug de agregação. O comparador v1 avaliava 8 pares humanos positivos e nenhum par negativo explícito, ocultando FP issue-level.

## Legacy vs Optimized

| Métrica | Legacy | Optimized | Delta |
|---|---:|---:|---:|
| Predicted relevant issues | 15 | 18 | +3 |
| Predicted non-relevant issues (binary) | 180 | 177 | -3 |
| TP | 5 | 7 | +2 |
| FP | 9 | 10 | +1 |
| FN | 2 | 0 | -2 |
| TN | 178 | 177 | -1 |
| Accuracy | 94.33% | 94.85% | 0.52% |
| Precision | 35.71% | 41.18% | 5.46% |
| Recall | 71.43% | 100.00% | 28.57% |
| Specificity | 95.19% | 94.65% | -0.53% |
| F1 | 47.62% | 58.33% | 10.71% |
| False positive rate | 4.81% | 5.35% | 0.53% |
| False negative rate | 28.57% | 0.00% | -28.57% |
| Balanced accuracy | 83.31% | 97.33% | 14.02% |

## Falsos positivos

- `Uniswap/v3-sdk#141` — Factory contract; insufficient_pattern_discussion / stage2_false_positive. CREATE2 address derivation is discussed without a creator/factory contract mechanism.
- `Consensys/gnark#1311` — Zero-knowledge proof; background_architecture / stage2_false_positive. ZK proof is library context; the PR focuses on hashing public inputs for calldata/recursion.
- `MetaMask/metamask-mobile#4441` — Self-Confirmed Transactions; insufficient_pattern_discussion / stage2_false_positive. The artifact is a wallet typed-data signing UI bug, not the selected catalog mechanism.
- `MoralisWeb3/react-moralis#142` — Identifier Registry; human_model_disagreement / stage2_false_positive. ENS resolution resembles a registry, but the reviewed annotation excludes it from the catalog pattern.
- `OpenZeppelin/openzeppelin-contracts#4084` — Role-based control; background_architecture / stage2_false_positive. AccessControl is only listed as a target of formal-verification tooling.
- `ProjectOpenSea/opensea-js#236` — Proxy contract; false_friend / stage2_false_positive. Wyvern user proxy is not the catalog's upgrade/delegation proxy mechanism.
- `matter-labs/foundry-zksync#358` — State initialization; false_friend / stage2_false_positive. Test-runner account migration/initialization is not blockchain state initialization as defined by the catalog.
- `paritytech/polkadot-sdk#4745` — State initialization; false_friend / stage2_false_positive. Runtime storage migration is not source-to-destination blockchain state initialization.
- `paritytech/substrate#9738` — Reverse Oracle; background_architecture / stage2_false_positive. Off-chain consumers are context; the PR changes event-record encoding/decoding.
- `trustwallet/assets#4740` — Reverse Oracle; human_model_disagreement / stage2_false_positive. A bot's payment-to-PR automation matches Reverse Oracle textually but is peripheral to the annotated PR topic.

## Falsos negativos

Optimized: nenhum. Legacy:
- `ethereum/pm#867` — stage2_false_negative; humano: Mortal.
- `graphprotocol/graph-node#3373` — stage1_miss; humano: Reverse Oracle.

## Diagnóstico por estágio

Os 10 FP optimized tiveram candidato Stage 1 e aceite Stage 2; a origem da decisão final divergente é Stage 2. Stage 1 sobregerou esses candidatos, mas não tomou a decisão final.

## Diagnóstico por pattern

- Reverse Oracle: FP=2, TP=1, proporção FP observada=66.67%.
- State initialization: FP=2, TP=0, proporção FP observada=100.00%.
- Factory contract: FP=1, TP=0, proporção FP observada=100.00%.
- Identifier Registry: FP=1, TP=0, proporção FP observada=100.00%.
- Proxy contract: FP=1, TP=1, proporção FP observada=50.00%.
- Role-based control: FP=1, TP=1, proporção FP observada=50.00%.
- Self-Confirmed Transactions: FP=1, TP=0, proporção FP observada=100.00%.
- Zero-knowledge proof: FP=1, TP=0, proporção FP observada=100.00%.

## Caso Time-Constrained Access

Optimized retrieval retained the pattern and selected the complete body/comments; compact and full definitions express the same time-window mechanism. The miss occurred inside Stage 1. Prompt/profile and stochastic generation both changed, so prompt_difference cannot be separated from model_variability in one observation.

Classificação: `not_determinable`. Retrieval, contexto e equivalência das definições foram confirmados; prompt vs variabilidade não são separáveis neste run.

## Limitações

- Apenas 7 positivos humanos; percentuais devem ser lidos com contagens absolutas.
- As anotações preservam decisões anteriores da LLM e não são gold standard independente/cego.
- Não existe universo completo de pares humanos negativos.
- Diagnósticos semânticos são hipóteses auditáveis, não relabeling automático.
- Uma execução por perfil não separa variação estocástica de efeitos do prompt.

## Recomendações

- Adjudicar manualmente os casos `human_model_disagreement`.
- Testar estabilidade/repetição em experimento separado, especialmente #3735.
- Investigar fronteiras de mecanismo em conjunto separado, sem derivar regras destes 200 casos.
- Manter gates issue-level como requisito permanente.

## Respostas às perguntas obrigatórias

1. Sim, optimized classificou 18 issues finais como relevantes.
2. As 18 são pós-agregação, não contagem intermediária.
3. Optimized: 18.
4. Legacy: 15.
5. Optimized recuperou 7 das 7 issues humanas yes.
6. Legacy recuperou 5 das 7.
7–15. Optimized TP/FP/FN/TN=7/10/0/177; precision/recall/F1/specificity/FPR=41.18%/100.00%/58.33%/94.65%/5.35%. Legacy=5/9/2/178; 35.71%/71.43%/47.62%/95.19%/4.81%.
16–21. Casos, patterns, evidências, estágios e diagnósticos estão nos CSVs; os FP optimized terminam em Stage 2.
22. Não foi encontrado bug na aggregation.
23. Sim: faltava avaliação issue-level, e os 187 negativos não entravam na precision anterior.
24. Ambos: havia lacuna de avaliação e há 10 decisões finais optimized divergentes da referência.
25. O miss #3735 ocorreu no Stage 1; contexto/definição não explicam e prompt vs variabilidade é indeterminável.
26. Optimized continua candidato experimental: recall=100%, precision issue-level=41,18%, ainda não aprovado.
27. Adjudicar desacordos e avaliar estabilidade/fronteiras em conjunto separado; nenhum ajuste semântico foi aplicado.
