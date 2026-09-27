# Llama local smoke sample

`llama_local_smoke.csv` is a deterministic 12-issue development sample copied
from `data/pilot/pilot_annotation_sample.csv`. Source text and prior annotation
columns were preserved; the source file was not modified.

The sample is intentionally mixed rather than accuracy-optimized:

- human-positive/reference cases: OpenZeppelin #3735, web3swift #103,
  ethereum/pm #867, and graph-node #3373;
- background architecture: OpenZeppelin #4084 and Consensys/gnark #1311;
- false-friend/mechanism-boundary cases: opensea-js #236,
  foundry-zksync #358, and Uniswap/v3-sdk #141;
- insufficient/peripheral pattern discussion: substrate #9738 and
  trustwallet/assets #4740;
- a simple negative/dependency case: js-ipfs #4085.

OpenZeppelin #3735 is included specifically to exercise retrieval of
`Time-Constrained Access` and the known Stage 1 miss. This sample is for smoke
testing, debugging, and manual inspection only. The 200-case pilot has already
influenced development decisions and must be treated as a development/reference
set, not an independent final evaluation set.
