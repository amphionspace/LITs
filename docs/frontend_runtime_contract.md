# Frontend runtime contract

The production English-Chinese inference path has one explicit boundary between
the text frontend and the acoustic model:

```text
raw text
  -> C++ TextNormalizer (data/zh or data/en)
  -> C++ mixed G2P + Chinese tone sandhi (data/en-zh-g2p)
  -> JSON-driven Text2Id (model_tokens.json)
  -> optional slicing of the encoded phoneme/ID result
  -> token_ids.jsonl
  -> LITs acoustic model
  -> Vocos
  -> wav
```

## Model input

Each non-empty JSONL line contains:

```json
{"phonemes":"...","token_ids":[1,2,3],"tone_ids":[0,0,0]}
```

`token_ids` is required and non-empty. `tone_ids` is either `null` or a
non-negative list with the same length. The launcher passes the frontend
`n_vocab` to the model process, which rejects a checkpoint/token-contract
mismatch before synthesis.

Each normalized input line runs C++ G2P and Text2Id exactly once. If the result
exceeds the model token budget, the launcher slices the already-produced
phonemes, token IDs, and tone IDs together. It does not call the frontend again
to probe candidate text chunks and does not keep a text-result cache. Chinese
tone sandhi therefore retains the full normalized-line context.

`phonemes.txt` is retained only for diagnosis, row alignment, and listening-test
traceability; the acoustic process does not tokenize it. With token-budget
splitting enabled, its rows are the final model chunks and align one-to-one with
`token_ids.jsonl`. The chunk manifest keeps the original normalized text for
merged output metadata.

## Dependency boundary

Frontend runtime dependencies:

- the unified C++ normalizer/G2P binary (or the equivalent C++ class/C API on device);
- ICU native libraries;
- profile JSON, rule JSON, dictionaries, Pinyin/Bopomofo mapping, and model token JSON
  shipped by the text-frontend submodule;
- Python standard library for desktop orchestration and JSON Text2Id.

Acoustic runtime dependencies remain `torch`, `numpy`, `soundfile`, `tqdm`, the
LITs model code, and the vendored Vocos implementation. These belong to the
model inference layer and are separate from the mobile C++ frontend migration.

The production entrypoints do not import `lits.text.language_cleaners`,
`frontend_rules`, `temp_cmu_g2p`, `ttsfrd`, Jieba, or Pypinyin. Those modules may
remain for training or explicit legacy-parity tools, but they are not on the
end-to-end runtime call graph.

The removed cache is only the Python frontend text-result cache. Long-lived
native CLI handles and the acoustic decoder/vocoder streaming caches are runtime
state with different responsibilities and remain in place.

## On-device replacement

The desktop launcher keeps one long-lived `tts_cli` process per profile. The
mobile implementation can replace that transport with long-lived native handles
without changing the resource or model-ID contracts:

```cpp
TextNormalizer zh_tn;
TextNormalizer en_tn;
TextNormalizer g2p;
zh_tn.Init("data/zh");
en_tn.Init("data/en");
g2p.Init("data/en-zh-g2p");

std::string normalized = has_hanzi(line)
    ? zh_tn.normalizeLine(line)
    : en_tn.normalizeLine(line);
std::string phonemes = g2p.normalizeLine(normalized);
// Apply model_tokens.json to obtain token IDs, then call the acoustic engine.
```

Normal pronunciation, exception, punctuation, dictionary, and token-inventory
updates are data updates. A genuinely new algorithm or native API still requires
updating the compiled library.
