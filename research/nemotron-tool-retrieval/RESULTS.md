# Nemotron tool-retrieval experiment

Date: 2026-09-20  
DayCare branch: `exp`  
GameTerm source revision: `9ed1b917`

## Result

Tool retrieval is a real, established design and multiple open-source
implementations already exist. A small GameTerm-specific probe found a viable
shape, but not a demonstrated accuracy improvement: retrieve ten candidates,
keep their original category order, retain the compact full category index, and
let Nemotron make the final call from the reduced schemas.

On the previously frozen 256-request suite, this reduced the serialized tool
definitions by **68.7%** while the trained candidate moved from **233/256** with
all 35 tools to **235/256** with ten. The paired change was **+0.78 percentage
points**, SD **27.99**, SE **1.75**, and bootstrap 95% CI **[-2.73, +4.30]**.
The interval includes zero, so this is parity, not evidence of better accuracy.
The untouched base similarly moved from **128/256** to **130/256**, CI **[-4.30,
+5.86]**.

All 32 plain-answer cases remained plain answers in both arms. All 223 calls
from the trained candidate had schema-valid arguments. No model, adapter, or
artifact was transferred to the MacBook Air.

## Existing work

This is usually called *tool retrieval*, *tool search*, or *dynamic tool
discovery*. The common design is a two-stage decision:

1. Search a catalog using the person's request and return a small candidate
   set.
2. Give the complete schemas for only those candidates to the tool-calling
   model, which makes the final choice and fills the arguments.

Relevant released systems include:

- [ToolBench](https://github.com/OpenBMB/ToolBench) includes retriever training
  code and an open-domain inference path that retrieves five APIs before tool
  use. Its corpus covers more than 16,000 APIs.
- [Semantic Tool Router](https://github.com/arunmm8335/semantic-tool-router) is
  a small MIT-licensed library that indexes tool descriptions, schemas, tags,
  examples, and permissions and injects a top-k set.
- [ToolRAG](https://github.com/antl3x/ToolRAG) is Apache-2.0 and provides
  embedding-based retrieval for OpenAI-format and MCP tools.
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) is a
  larger Apache-2.0 routing layer with tool filtering. It is active and
  production-oriented, but much broader than GameTerm needs.
- [Tool2Vec](https://github.com/SqueezeAILab/Tool2Vec) and
  [HYSET](https://github.com/stormwther18/HYSET) are MIT-licensed research
  implementations for learned and multi-tool-set retrieval.
- [Tool-REX](https://github.com/EIT-NLP/Tool-REX) is an Apache-2.0 framework
  built around richer tool documents and dedicated retriever/reranker models.
  Its paper reports that fields such as function, when-to-use, limitations, and
  tags can improve retrieval, while adding every field is not always best.

Anthropic also ships a provider-side version of this pattern. Its
[tool-search documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
describes regex and BM25 search over names, descriptions, argument names, and
argument descriptions, followed by loading up to five complete definitions.
That service is not GameTerm's implementation, but its documented architecture
closely matches this proposal.

## Method

The probe indexed the 35 complete categorized GameTerm definitions. Each search
document contained only query-independent metadata: tool name, category,
description, argument names, argument descriptions, and enum values. It did not
use suite requests or answer keys to build or tune the index.

Four retrieval rankings were compared on the 224 tool-needed requests:

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| BM25 | 50.0% | 70.5% | 86.2% | 98.2% |
| BM25, category prefix removed | 49.6% | 70.5% | 86.2% | 98.2% |
| all-MiniLM-L6-v2 cosine | 38.4% | 71.4% | 80.4% | 93.3% |
| BM25 + dense reciprocal-rank fusion | **54.5%** | **85.3%** | **94.2%** | **98.7%** |

The dense arm used the Apache-2.0
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
encoder with mean pooling and normalized cosine similarity. With only 35 tools,
the vectors fit in memory; a vector database would add persistence and
approximate-nearest-neighbor search, not better ranking by itself.

The end-to-end arms used the same full GameTerm system text, permission ledger,
temperature zero, `enable_thinking: false`, real schemas, and real schema
validation as the earlier experiment. No training occurred.

## End-to-end results

| Model and presentation | Score | No tool | Valid calls |
|---|---:|---:|---:|
| Base, all 35 tools | 128/256 | 32/32 | 104/104 |
| Base, ranked top 5 | 119/256 | 32/32 | 94/94 |
| Base, stable top 10 | 130/256 | 32/32 | 107/107 |
| Candidate, all 35 tools | 233/256 | 32/32 | 223/223 |
| Candidate, ranked top 5 | 212/256 | 32/32 | 213/215 |
| Candidate, stable top 10 | **235/256** | **32/32** | **223/223** |

The ranked top-five variant changed two things at once: it narrowed the set and
reordered tools by retrieval score. It significantly hurt the candidate:
**-8.20 points**, SD **38.23**, SE **2.39**, 95% CI **[-12.89, -3.52]**. This
shows that high retrieval recall alone is insufficient. The model was trained
against a stable categorized layout, and a small dynamically reordered menu
changed that learned setting.

The stable top-ten variant preserved the original order among selected tools
and kept the compact category-to-name map for all 35 tools. Its candidate
result by group was:

| Group | All 35 | Stable top 10 |
|---|---:|---:|
| calculate | 30/32 | 31/32 |
| terminal | 32/32 | 30/32 |
| files | 32/32 | 31/32 |
| apps | 15/32 | 20/32 |
| media | 32/32 | 29/32 |
| windows | 31/32 | 31/32 |
| web | 29/32 | 31/32 |
| no tool | 32/32 | 32/32 |

It produced 11 paired gains and nine losses. Three of its 21 errors were hard
retrieval misses; the other 18 occurred even though an acceptable tool was
available. The remaining bottleneck is therefore mostly the model's final
choice, especially among apps and nearby window tools.

## Interpretation

The user's core idea is sound. Retrieval can remove about two thirds of the
schema text without measurably hurting this candidate, and the hybrid ranking
is substantially better than either sparse or dense retrieval alone. The
category label alone made little difference to BM25 because it is one repeated
word per tool; categories matter more as stable organization for Nemotron than
as lexical search terms.

A vector database is unnecessary at 35 tools. The smallest useful GameTerm
design is an in-memory catalog with BM25 and small embeddings, reciprocal-rank
fusion, stable category order, and a conservative top-ten result. The model
should still make the final tool/no-tool decision. Directly executing the
retriever's top result would be unsafe and inaccurate: hybrid top-1 recall was
only 54.5%.

This suite was reused to compare top-five and top-ten designs, so this is an
exploratory result. It must not be presented as a fresh confirmatory holdout.
Before adoption, freeze a new suite and predeclare the stable top-ten design,
the full-tool control, retention requirements, and a latency target. A stronger
next arm would train the model on varying stable shortlists plus a `search_tools`
call, since the current adapter learned only the fixed 35-tool presentation.

## Artifacts

Raw outputs remain outside Git under
`~/storage/daycare-runs/nemotron-tool-retrieval-001/`. Important SHA-256 values:

- retrieval rankings: `1685746b03de8dfd6d052f5a60fcfbae14bc13b9dd211933d864522a102c6636`
- base stable top ten: `c674d8442583c85893ee562a7ab0553229e2dfe4ea2d1096af366fb8d861e413`
- candidate stable top ten: `0e0b3d53c8627ef1e95d8a049359881f45931267bd2b242775502064aac49d2f`

