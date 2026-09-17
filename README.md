# MAWILE

MAWILE (**Multi-Axis Workbench for Inspecting LLM Evaluators**) is a black-box
reliability audit for configured LLM judges. It treats the
judge prompt, rubric, model, decoding settings, and output schema as one
measurement instrument, then estimates where the resulting verdicts are noisy,
fragile, or worth human review.

## Quick Start

```bash
python -m pip install -e '.[dev]'
mawile run data/demo/configs/mock_judge.yaml
mawile ui
```

Generated artifacts are written under `runs/<timestamp>/`.

The new local HTML UI starts with `mawile ui` and serves
<http://127.0.0.1:8000> by default.
Optional agent overrides use one rule in the UI: blank means inherit from the
perturbation generator; `{}` means explicitly empty decoding parameters.

## CLI Workflows

The command line supports the same audit spine as the UI:

```bash
mawile configs
mawile catalog --validation
mawile inspect data/demo/configs/mock_judge.yaml --previews
mawile run data/demo/configs/mock_judge.yaml --progress
mawile runs
mawile report
```

LLM-backed planning helpers are available when a non-`mock` suggester model and
the API key required by its selected provider are configured:

```bash
mawile plan path/to/config.yaml --output planned.yaml
mawile suggest-custom path/to/config.yaml --target item.output --format yaml
```

Most inspection commands also support `--format json` for scripts.


## Environment

Create or update the conda environment from the repo root:

```bash
conda env create -f environment.yml
conda activate mawile
python -m pip install -e .
```

Python dependencies are declared in `pyproject.toml`; `requirements.txt` installs the development extra from that single source. Copy `.env.example` to
`.env` and fill in the key for each provider you use (`OPENAI_API_KEY`,
`FIREWORKS_API_KEY`, or `GEMINI_API_KEY`). MAWILE loads `.env` automatically
from the directory where you start the command; if you prefer to source it in
your shell, use `export OPENAI_API_KEY=...` or
`set -a; source .env; set +a` so child processes inherit the value.

```bash
cp .env.example .env
# Fill the provider keys used by your config in .env
mawile run data/demo/configs/openai_judge.example.yaml
```

## Repo Layout

```text
data/                   Dataset bundles with raw files, items, samples, and configs
runs/                   Timestamped audit run directories
scripts/                Data conversion and maintenance scripts
src/mawile/             Reusable audit package
```

Dataset bundles keep the inputs for an audit together:

```text
data/<dataset>/
  raw_data/             Original source files, when available
  items.jsonl           Canonical audit items
  samples/              Optional smaller slices
  configs/              Dataset-specific judge configs
  README.md             Optional dataset notes
```

Each audit run is self-contained:

```text
runs/<timestamp>/
  run.json
  report.md
  item_risks.csv
  config.resolved.yaml
  suite.json
  perturbations.jsonl
  judge_results.jsonl
  # run.json also includes attempted_perturbations, operator_manifest,
  # and perturbation_ledger for rejected/skipped perturbation transparency.
  traces/
    llm_calls.jsonl
    generation_calls.jsonl
    validation_calls.jsonl
    judge_calls.jsonl
  logs/
    events.jsonl
```

The `src/mawile` package is split by responsibility:

- `config.py` loads YAML audit configs
- `schemas.py` defines canonical items, results, perturbations, and risks
- `judges/` contains black-box judge runners
- `perturbations/` declares every perturbation operator in `registry.py`;
  deterministic operators build offline, while LLM-backed operators run through
  the perturbation agent under `perturbations/generated/`
- `validation/` runs independent, dimension-aware admissibility checks on
  semantic perturbations (including rule-generated proposals) and gates them via
  `validity_status`
- `metrics/` computes noise, flip-risk, signed-shift attribution,
  judge correctness against gold labels, directional degradation detection,
  bootstrap confidence intervals, review ranking, and cost
- `reporting/` writes self-contained run directories
- `suites.py` validates and loads exact saved probe suites for judge-only replay
- `pipeline/` orchestrates an end-to-end audit run
- `web/` serves the local FastAPI/Jinja HTML UI

## Perturbations

Choose explicit operator names in `audit.perturbation_operators` and
`audit.directional_perturbation_operators`. An empty list means no operators in
that category. With both empty and no enabled custom probes, the run measures
only the repeated baseline. Unknown or unsupported selections are errors.

`mawile plan config.yaml --output planned.yaml` produces an editable selection.
The suggested built-in lists replace the previous lists; manual edits made
afterward are authoritative. `audit.plan_operators: true` performs planning when
execution starts. Planning never occurs merely by constructing an audit object.
Planner examples use a recorded seed; saving a plan freezes the operator choices,
but only saved variants preserve the actual LLM rewrites.

There are two current expected relations:

- **Invariant** (`same_verdict`): preserve the semantic judgment. Numeric judges
  use the configured score tolerance; binary judges compare classes; pairwise
  judges compare stable candidate identities.
- **Directional** (`worse_verdict`): introduce a validated degradation that the
  judge should detect. Directional metrics are separate from invariant flip risk.

**Pairwise audits support invariant probes only.** Directional built-in and custom
probes are unavailable. Position swap is an ordinary invariant: a candidate chosen
at A before the swap should still be chosen when displayed at B. Changing the
display label does not itself constitute a verdict flip. Historical reports may
retain their old equivariant measurements; these are not comparable by simple
relabeling to the new repeated-verdict invariant estimator.

### Directional perturbations

Directional probes are opt-in for pointwise judges and can be selected manually or by the Perturbation Suggester:

- `agent_output_partial_completion` (Rule proposal, semantic validation): remove a final output chunk, then check that a judged requirement or quality property was degraded.
- `agent_output_format_violation` (LLM, `format_degradation`): break exactly one explicit output-format requirement that the current output satisfies, without changing substantive content.
- `agent_input_unmet_requirement_injection` (LLM, `unmet_requirement`): add one plausible requirement to the input that the unchanged output does not satisfy.
- `agent_output_unsupported_claim_insertion` (LLM, `directional_degradation`): add one plausible unsupported claim.
- `agent_output_factual_inconsistency_insertion` (LLM, `directional_degradation`): introduce one small factual or reasoning inconsistency.
- `agent_output_requirement_omission` (LLM, `directional_degradation`): remove or weaken one requirement-satisfying part of the output.
- `agent_output_over_refusal` (LLM, `refusal_degradation`): turn an answerable response into an unnecessary refusal; inapplicable items are recorded as generation skips.

### Invariant perturbations

The registered invariant operators, grouped by component. `Type` is `Rule` (offline
string transform with structural or semantic validation) or `LLM` (perturbation-agent rewrite). `Check` is
the admissibility check applied before a variant counts (`—` for
deterministic ones). [`registry.py`](src/mawile/perturbations/registry.py) is the
single source of truth.

#### `judge_prompt_*` — the judge's own instructions (`touches: judge_wording`)

| Operator | Type | Check | What it does |
| --- | --- | --- | --- |
| `judge_prompt_instruction_order` | Rule | — | Reorder the assembled prompt/rubric/output blocks without rewriting any of them. |
| `judge_prompt_reasoning_style` | Rule | `judge_prompt_preserve` | Append a reasoning-style directive: direct, step-by-step, or criterion-by-criterion. |
| `judge_prompt_role_framing` | LLM | `judge_prompt_preserve` | Change the evaluator persona, leaving the task fixed. |
| `judge_prompt_paraphrase` | LLM | `judge_prompt_preserve` | Reword the whole prompt, instructions preserved. |

#### `judge_rubric_*` — the scoring rubric (all `touches: judge_wording`)

| Operator | Type | Check | What it does |
| --- | --- | --- | --- |
| `judge_rubric_criterion_reorder` | Rule | `judge_rubric_preserve` | Reorder intact explicitly delimited rubric blocks; ambiguous prose is skipped. Flat numeric score rows may use structural validation. |
| `judge_rubric_paraphrase` | LLM | `judge_rubric_preserve` | Reword each criterion; meaning, ordering, and thresholds preserved. |

#### `agent_input_*` — the agent's input (part of `x`)

| Operator | Type | Touches | Check | What it does |
| --- | --- | --- | --- | --- |
| `agent_input_minor_typo_noise` | LLM | surface | `typo_preserve` | Inject a few natural prose typos while preserving exact-syntax content. |
| `agent_input_paraphrase` | LLM | lexical | `meaning_preserve` | Reword the input; meaning and requirements preserved. |
| `agent_input_distractor_insertion` | LLM | lexical | `sample_plausibility` | Add plausible but irrelevant sentences. |
| `agent_input_context_compression` | LLM | facts | `facts_preserve` | Drop only redundant context, keeping every required fact. |

#### `agent_output_*` — the graded output (part of `y`)

| Operator | Type | Touches | Check | What it does |
| --- | --- | --- | --- | --- |
| `agent_output_format_conversion` | LLM | format | `same_answer` | Convert between prose, bullets, numbered list, or table. |
| `agent_output_politeness_tone_shift` | LLM | tone | `same_answer` | Change politeness or tone; substance unchanged. |
| `agent_output_uncertainty_calibration` | LLM | confidence | `same_answer` | Make the same answer sound more or less certain. |
| `agent_output_verbosity_shift` | LLM | length | `same_answer` | Make the output shorter or longer; same answer. |

#### `agent_paired_*` — edits kept coherent across `x` and `y`

| Operator | Type | Touches | Check | What it does |
| --- | --- | --- | --- | --- |
| `agent_paired_position_swap` | Rule | candidate_position | — | Swap A/B display positions while preserving candidate IDs and content; compare the same candidate identity as an invariant verdict. |
| `agent_paired_entity_rename` | LLM | identity | `pair_coherence` | Rename named entities consistently across every section — the bias keystone. |

## Config reference

A run config is a YAML file with nine top-level blocks (`providers`, `judge`,
`data`, `audit`, `output`, `perturbation_agent`, `validator_agent`,
`planner_agent`, `summary_agent`). Only `judge` and `data` are required; the rest
fall back to the defaults below.
The canonical definitions live in [`schemas.py`](src/mawile/schemas.py).

### `providers:` — named OpenAI-compatible endpoints

Every LLM role uses the OpenAI SDK's Chat Completions interface. `openai`,
`fireworks`, and `gemini` are built in with their standard base URLs and the
environment variables shown above. The map is optional and can add a provider
or replace a built-in profile without changing MAWILE code:

```yaml
providers:
  local_gateway:
    api_key_env: LOCAL_GATEWAY_API_KEY
    base_url: http://localhost:8000/v1

validator_agent:
  provider: local_gateway
  model: independent-validator-model
```

Only the environment-variable name is serialized in run artifacts; API-key
values are read at runtime. See
[`cross_provider_validation.example.yaml`](data/demo/configs/cross_provider_validation.example.yaml)
for an OpenAI judge, Fireworks perturbation generator, and Gemini validator.

### `judge:` — the judge under test

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | str | `openai` | Built-in or config-defined provider profile used for the judge. |
| `model` | str | _required_ | Model identifier passed to the runner. |
| `prompt_template` | str | _required_ | Prompt template the judge is given. |
| `rubric` | str | _required_ | Rubric text injected into the prompt. |
| `decoding_params` | map | `{}` | Runner decoding options (e.g. temperature, top_p). |
| `output_schema` | map / str / null | `null` | Expected structured-output schema; surfaced in the prompt when set. |
| `threshold` | float / null | `null` | Optional pass/fail cutoff for scalar and ordinal verdicts; the passing side follows `score_direction`. |
| `score_min`, `score_max` | float / null | `null` | Inclusive valid score range; set both or neither. |
| `score_direction` | enum | `higher_is_better` | Whether larger or smaller scores represent better quality. |
| `invariant_tolerance` | float ≥ 0 | `0` | Maximum scalar/ordinal change still counted as invariant. |
| `gold_tolerance` | float ≥ 0 / null | `null` | Optional scalar gold-error tolerance used to define tolerance accuracy. |
| `output_type` | enum | `binary` | Verdict shape: `binary`, `ordinal`, `scalar`, `pairwise`, or `structured`. |

### `data:` — input items

Pointwise items contain exactly two judged transcript elements: the complete
`input` and complete `output`. Any instructions belong in `input`; any visible
reasoning or execution details belong in `output`. Pairwise items replace the
pointwise value inside `output` with stable `candidates` plus an explicit
`position_to_candidate` mapping for A and B.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `items_path` | path | _required_ | JSONL file of items to audit. |
| `context_path` | path / null | `null` | Optional explicit text, Markdown, or JSON description included in dataset-aware suggestion prompts. Documentation is not auto-discovered. |
| `input_field` | str | `input` | Column mapped to the complete agent input, including instructions. |
| `output_field` | str | `output` | Column mapped to the complete agent output; pairwise structure is nested here. |
| `gold_field` | str / null | `gold_label` | Column with the gold label; set `null` if items have none. |

Binary gold is a normalized semantic class; pairwise gold is a stable preferred
candidate ID; scalar gold is numeric. Reports use class/pairwise accuracy, or
scalar MAE, Spearman rank correlation, and optional tolerance accuracy.

### `audit:` — what to run and how to score it

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `perturbation_operators` | list | `[]` (none) | Explicit operator names to build; empty means no built-in invariant probes. See [Perturbations](#perturbations). |
| `directional_perturbation_operators` | list | `[]` (none) | Explicit allow-list of directional degradation operators. These probes intentionally make items worse and are reported separately from invariant flip-risk. |
| `custom_perturbations` | list | `[]` | User-defined LLM-backed perturbations. Enabled custom perturbations always run, regardless of `perturbation_operators`. Each entry has `name`, `target`, `instruction`, optional `expected_effect` (`same_verdict` or `worse_verdict`), and optional `enabled`; targets are `judge.prompt_template`, `judge.rubric`, `item.input`, or `item.output`. |
| `plan_operators` | bool | `false` | Auto-route the invariant and directional allow-lists with the Perturbation Suggester instead of hand-picking them. The suggester uses `planner_agent` when configured and otherwise defaults to `perturbation_agent`, independent of the judge. The resulting built-in lists replace the previous selections and remain editable. Needs a non-`mock` suggester model and its provider key; a failed suggestion call aborts the run. |
| `repeats` | int ≥ 1 | `10` | Times **every** judge call is repeated — the original (which sets the baseline and observed repeat noise) and each perturbation variant — so a variant's flip rate is measured on equal footing with the baseline noise. |
| `num_workers` | int ≥ 1 | `1` | Worker threads issuing perturbation-generation and judge calls concurrently. `1` runs each phase sequentially; higher values overlap per-call API latency. Results are order-stable regardless of this value. |
| `scalar_delta` | float > 0 | `1.0` | Scalar margin for the boundary-proximity signal: an item's proximity rises from 0 at `2·scalar_delta` away from `judge.threshold` to 1 at the threshold. Used only as review context; item risk is ranked by raw invariant flip risk. |
| `top_k_routes` | int ≥ 1 | `10` | Number of highest flip-risk items listed in the review section of the report. |
| `bootstrap_samples` | int ≥ 100 | `1000` | Resamples used for bootstrap confidence intervals. |
| `bootstrap_seed` | int | `42` | RNG seed for reproducible bootstrap intervals. |

Example custom perturbation:

```yaml
audit:
  custom_perturbations:
    - name: Legal wording shift
      target: item.input
      instruction: Rewrite the input in dense legal language while preserving every fact and requirement.
      expected_effect: same_verdict
      enabled: true
```

### `output:` — where results land

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `runs_dir` | path | `runs` | Directory under which each run's self-contained report is written. |

### `perturbation_agent:` — the model that generates LLM-backed dimensions

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | str | `openai` | Built-in or config-defined provider profile for generation. |
| `model` | str | `mock` | Model that generates LLM-backed dimensions. Must be independent of the judge under test. `mock` supports offline demos; explicitly selected LLM probes require a configured generator. |
| `decoding_params` | map | `{}` | Decoding options forwarded to the perturbation-agent runner. |

### `validator_agent:` — optional override for admissibility checks

When omitted, the validator inherits the perturbation agent's provider, model,
and decoding parameters. Semantic probes require an available semantic validator;
structural rule checks do not substitute for it. Freeze the validator explicitly
when comparing generation models.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | str / null | `null` | Provider override. Setting it also requires an explicit validator `model`. |
| `model` | str / null | `null` | Model used only for LLM admissibility validation. |
| `decoding_params` | map / null | `null` | Decoding options forwarded to the validator; `null` inherits the perturbation-agent settings. |

### `planner_agent:` — optional Perturbation Suggester override

When omitted, the suggester inherits the perturbation agent's provider, model,
and decoding parameters.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | str / null | `null` | Provider override. Setting it also requires an explicit suggester `model`. |
| `model` | str / null | `null` | Model used only for perturbation suggestions when `audit.plan_operators` is true. Set this to route with the suggester even when `perturbation_agent.model` is `mock`. |
| `decoding_params` | map / null | `null` | Decoding options forwarded to the suggester; `null` inherits the perturbation-agent settings. |
| `sampling_seed` | int | `42` | Seed for the planner's sampled dataset examples. |
| `dataset_context_max_chars` | int ≥ 1000 | `40000` | Maximum characters of dataset statistics, explicit context, and sampled items sent to either suggestion feature. Small datasets are included fully; larger ones use a seeded sample recorded in planner metadata. |

### `summary_agent:` — optional override for report summaries

When omitted, the summary agent inherits the perturbation agent's provider,
model, and decoding parameters. If the resolved model is `mock` or its provider
key is absent, the report uses the deterministic summary.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | str / null | `null` | Provider override. Setting it also requires an explicit summary `model`. |
| `model` | str / null | `null` | Model used only for the report interpretation summary. |
| `decoding_params` | map / null | `null` | Decoding options forwarded to the summary agent; `null` inherits the perturbation-agent settings. |

### Cost & pricing

Pricing is **not** a config field. The run report prices recorded token usage
from every LLM-backed phase: judge calls, operator planning, perturbation
generation, equivalence validation, and report generation. The cost artifact
contains aggregate totals and a `by_category` breakdown with calls, input
tokens, cached-read tokens, cache-write tokens, output tokens, and USD for each
active phase.

Costs use the central provider-and-model table in
[`pricing.py`](src/mawile/pricing.py) (USD per 1M input, cached-input, and output
tokens, plus cache writes where the provider charges separately). Edit or extend
that table for the models you run. When a call lacks a
price entry or token usage, the report marks pricing as incomplete and presents
the known USD subtotal as a lower bound. Models without price entries are also
listed under `unpriced_models`. The `mock` runner makes no paid model calls and
reports no token usage.