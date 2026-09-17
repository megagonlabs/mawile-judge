from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OutputType(StrEnum):
    BINARY = "binary"
    ORDINAL = "ordinal"
    SCALAR = "scalar"
    PAIRWISE = "pairwise"
    STRUCTURED = "structured"


class ScoreDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


BUILTIN_PROVIDER_NAMES = frozenset({"openai", "fireworks", "gemini"})


class ExpectedRelation(StrEnum):
    INVARIANT = "invariant"
    DIRECTIONAL = "directional"
    EQUIVARIANT = "equivariant"


class PerturbationExpectedEffect(StrEnum):
    SAME_VERDICT = "same_verdict"
    WORSE_VERDICT = "worse_verdict"
    SAME_CANDIDATE = "same_candidate"


class PerturbationFamily(StrEnum):
    NOISE = "noise"
    JUDGE_PROMPT = "judge_prompt"
    JUDGE_RUBRIC = "judge_rubric"
    AGENT_INPUT = "agent_input"
    AGENT_OUTPUT = "agent_output"
    AGENT_PAIRED = "agent_paired"


CustomPerturbationTarget = Literal[
    "judge.prompt_template",
    "judge.rubric",
    "item.input",
    "item.output",
]


class ProviderConfig(BaseModel):
    """Connection profile for an OpenAI-SDK-compatible model endpoint.

    Only the environment-variable *name* is stored in config and artifacts; the
    credential value is resolved at runtime and is never serialized.
    """

    model_config = ConfigDict(extra="forbid")

    api_key_env: str = Field(
        description="Name of the environment variable containing this provider's API key."
    )
    base_url: str = Field(
        description="Absolute HTTP(S) base URL passed to the OpenAI client."
    )

    @field_validator("api_key_env")
    @classmethod
    def _valid_api_key_env(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(
                "api_key_env must be an environment-variable name, not a key value or ${...} reference"
            )
        return value

    @field_validator("base_url")
    @classmethod
    def _valid_base_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        return value


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default="openai",
        description=(
            "Named provider profile used for the judge. Built-ins are openai, "
            "fireworks, and gemini; additional profiles may be declared under providers."
        ),
    )

    model: str = Field(
        description=(
            "Model identifier for the judge under test. 'mock' uses the "
            "deterministic local runner; any other value is passed unchanged to "
            "the selected provider."
        )
    )
    prompt_template: str = Field(
        description="Instruction text the judge receives, before the rubric and output instructions."
    )
    rubric: str = Field(description="Scoring rubric injected into the judge prompt.")
    decoding_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Chat Completions options forwarded to the runner (e.g. temperature, "
            "top_p, max_completion_tokens, or reasoning_effort)."
        ),
    )
    output_schema: dict[str, Any] | str | None = Field(
        default=None,
        description="Optional structured-output schema; rendered into the prompt when set.",
    )
    threshold: float | None = Field(
        default=None,
        description=(
            "Optional pass/fail cutoff for scalar and ordinal verdicts. Which "
            "side passes is determined by score_direction. Leave unset when the "
            "numeric score should not be thresholded."
        ),
    )
    score_min: float | None = Field(
        default=None,
        description="Inclusive minimum valid score for scalar or ordinal outputs.",
    )
    score_max: float | None = Field(
        default=None,
        description="Inclusive maximum valid score for scalar or ordinal outputs.",
    )
    score_direction: ScoreDirection = Field(
        default=ScoreDirection.HIGHER_IS_BETTER,
        description="Whether larger or smaller numeric scores represent better quality.",
    )
    invariant_tolerance: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Maximum absolute scalar/ordinal score change still counted as invariant. "
            "Binary and pairwise outputs use semantic identity instead."
        ),
    )
    gold_tolerance: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional absolute-error tolerance for scalar/ordinal gold accuracy. "
            "MAE and rank correlation are reported regardless."
        ),
    )
    output_type: OutputType = Field(
        default=OutputType.BINARY,
        description="Verdict shape the parser expects: binary, ordinal, scalar, pairwise, or structured.",
    )
    prompt_block_order: list[Literal["prompt", "rubric", "output"]] = Field(
        default_factory=lambda: ["prompt", "rubric", "output"],
        description=(
            "Internal assembly order for the judge prompt blocks. Normal configs use "
            "the default; prompt-order perturbations override this without rewriting "
            "the prompt, rubric, or output instruction text."
        ),
    )

    @model_validator(mode="after")
    def _validate_score_semantics(self) -> "JudgeConfig":
        if (self.score_min is None) != (self.score_max is None):
            raise ValueError("score_min and score_max must be set together")
        if self.score_min is not None and self.score_max is not None:
            if self.score_min >= self.score_max:
                raise ValueError("score_min must be less than score_max")
            if self.threshold is not None and not self.score_min <= self.threshold <= self.score_max:
                raise ValueError("threshold must lie within score_min and score_max")
        return self


class PerturbationAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default="openai",
        description="Named provider profile for LLM-backed perturbation generation.",
    )

    model: str = Field(
        default="mock",
        description=(
            "Model that generates LLM-backed perturbations and runs the equivalence "
            "checks. Must be independent of the judge under test -- using the judge's "
            "own model to vet its probes is circular. 'mock' cannot generate or "
            "validate LLM-backed operators, so explicitly selecting one is invalid."
        ),
    )
    decoding_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Decoding options forwarded to the perturbation-agent runner.",
    )


def _require_model_for_provider_override(
    role: str,
    provider: str | None,
    model: str | None,
) -> None:
    """Keep inherited-role provider overrides independently executable."""

    if provider is not None and model is None:
        raise ValueError(f"{role}.provider requires {role}.model")


class PlannerAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = Field(
        default=None,
        description=(
            "Optional provider override for the Perturbation Suggester. When unset, "
            "the suggester inherits perturbation_agent.provider."
        ),
    )

    model: str | None = Field(
        default=None,
        description=(
            "Optional model override for the Perturbation Suggester. When unset, "
            "the suggester uses perturbation_agent.model."
        ),
    )
    decoding_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional decoding options for the Perturbation Suggester. When unset, "
            "the suggester uses perturbation_agent.decoding_params."
        ),
    )
    dataset_context_max_chars: int = Field(
        default=40000,
        ge=1000,
        description=(
            "Maximum characters of dataset statistics, documentation, and sampled "
            "items included in Perturbation Suggester prompts."
        ),
    )
    sampling_seed: int = Field(
        default=42,
        description=(
            "Seed used when the suggester samples oversized datasets. The realized "
            "seed and item IDs are recorded with the planning trace."
        ),
    )

    @model_validator(mode="after")
    def _provider_override_has_model(self) -> "PlannerAgentConfig":
        _require_model_for_provider_override("planner_agent", self.provider, self.model)
        return self


class SummaryAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = Field(
        default=None,
        description=(
            "Optional provider override for the report summary agent. When unset, "
            "the summary agent inherits perturbation_agent.provider."
        ),
    )

    model: str | None = Field(
        default=None,
        description=(
            "Optional model override for the report summary agent. When unset, "
            "the summary agent uses perturbation_agent.model."
        ),
    )
    decoding_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional decoding options for the report summary agent. When unset, "
            "the summary agent uses perturbation_agent.decoding_params."
        ),
    )

    @model_validator(mode="after")
    def _provider_override_has_model(self) -> "SummaryAgentConfig":
        _require_model_for_provider_override("summary_agent", self.provider, self.model)
        return self


class ValidatorAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = Field(
        default=None,
        description=(
            "Optional provider override for LLM admissibility validation. When unset, "
            "the validator inherits perturbation_agent.provider."
        ),
    )

    model: str | None = Field(
        default=None,
        description=(
            "Optional model override for LLM admissibility validation. When unset, "
            "the validator uses perturbation_agent.model."
        ),
    )
    decoding_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional decoding options for the validator. When unset, the validator "
            "uses perturbation_agent.decoding_params."
        ),
    )

    @model_validator(mode="after")
    def _provider_override_has_model(self) -> "ValidatorAgentConfig":
        _require_model_for_provider_override("validator_agent", self.provider, self.model)
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items_path: Path = Field(
        description="Path to the JSONL file of items to audit (resolved relative to the config file)."
    )
    context_path: Path | None = Field(
        default=None,
        description=(
            "Optional text, Markdown, or JSON file containing curated dataset context "
            "for the Perturbation Suggester; resolved relative to the config file."
        ),
    )
    input_field: str = Field(
        default="input",
        description="Row key mapped to the complete agent input, including instructions.",
    )
    output_field: str = Field(
        default="output",
        description=(
            "Row key mapped to the complete agent output; pairwise candidate "
            "structure is nested within it."
        ),
    )
    gold_field: str | None = Field(
        default="gold_label",
        description="Row key holding the gold label, or null when items have no gold labels.",
    )


class CustomPerturbationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description=(
            "Human-readable name for a user-defined invariant perturbation. The "
            "name is used to derive a stable operator identifier in run artifacts."
        )
    )
    target: CustomPerturbationTarget = Field(
        description=(
            "Field rewritten by the custom perturbation. Supported targets are "
            "judge.prompt_template, judge.rubric, item.input, and item.output."
        )
    )
    instruction: str = Field(
        description=(
            "Prompt sent to the perturbation agent describing how to rewrite the "
            "selected field according to the intended relationship."
        )
    )
    expected_effect: PerturbationExpectedEffect = Field(
        default=PerturbationExpectedEffect.SAME_VERDICT,
        description=(
            "Intended relationship for this custom perturbation. same_verdict makes "
            "it an invariant check; worse_verdict makes it a directional degradation."
        ),
    )
    enabled: bool = Field(
        default=True,
        description="Whether this custom perturbation should run.",
    )

    @field_validator("name", "instruction")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-empty")
        return stripped

    @field_validator("expected_effect")
    @classmethod
    def _custom_expected_effect_is_supported(
        cls, value: PerturbationExpectedEffect
    ) -> PerturbationExpectedEffect:
        if value not in {
            PerturbationExpectedEffect.SAME_VERDICT,
            PerturbationExpectedEffect.WORSE_VERDICT,
        }:
            raise ValueError("custom perturbations support same_verdict or worse_verdict")
        return value


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perturbation_operators: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit allow-list of perturbation operators to build, named "
            "'<component>_<operator>' (e.g. 'judge_prompt_instruction_order', "
            "'agent_input_paraphrase'). This list is the complete built-in selection: "
            "an empty list runs no perturbation probes (baseline only). LLM-backed "
            "operators additionally require a non-mock 'perturbation_agent.model' "
            "plus an API key; all selected operators remain subject to their guards "
            "and validation requirements."
        ),
    )
    directional_perturbation_operators: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit allow-list of directional perturbation operators to build. "
            "Directional operators intentionally degrade the item and default to "
            "'worse_verdict'; an empty list means none are run."
        ),
    )
    custom_perturbations: list[CustomPerturbationConfig] = Field(
        default_factory=list,
        description=(
            "User-defined LLM-backed perturbations. Each enabled custom perturbation "
            "is an explicitly selected probe, independent of the built-in operator list."
        ),
    )
    plan_operators: bool = Field(
        default=False,
        description=(
            "Auto-route the operator selection with the Perturbation Suggester. The suggester "
            "reads the judge under test and replaces both operator lists with its selected "
            "invariant and directional probes. Requires a non-mock suggester model "
            "plus an API key; a failed suggestion call aborts the run rather than guessing."
        ),
    )
    repeats: int = Field(
        default=10,
        ge=1,
        description=(
            "Times every judge call is repeated -- both the original (which sets the "
            "noise floor) and each perturbation variant -- so a "
            "variant's flip rate is measured on equal footing with the baseline noise."
        ),
    )
    num_workers: int = Field(
        default=1,
        ge=1,
        description=(
            "Worker threads issuing perturbation-generation and judge calls "
            "concurrently. 1 runs each phase sequentially; higher values overlap "
            "the per-call API latency."
        ),
    )
    scalar_delta: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Scalar margin for the boundary-proximity signal: proximity rises "
            "from 0 at 2*scalar_delta away from judge.threshold to 1 at the threshold. "
            "Used only as review context; item risk is ranked by raw invariant flip risk."
        ),
    )
    top_k_routes: int = Field(
        default=10,
        ge=1,
        description="Number of highest-risk items listed in the routing section of the report.",
    )
    bootstrap_samples: int = Field(
        default=1000, ge=100, description="Resamples used for bootstrap confidence intervals."
    )
    bootstrap_seed: int = Field(
        default=42, description="RNG seed for reproducible bootstrap intervals."
    )

    @model_validator(mode="after")
    def _custom_perturbation_names_are_unique(self) -> "AuditConfig":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for perturbation in self.custom_perturbations:
            if not perturbation.enabled:
                continue
            key = perturbation.name.casefold()
            if key in seen:
                duplicates.add(perturbation.name)
            seen.add(key)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate custom perturbation names: {names}")
        return self


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs_dir: Path = Field(
        default=Path("runs"),
        description="Directory under which each run's self-contained report directory is written.",
    )


class AuditRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judge: JudgeConfig
    data: DataConfig
    audit: AuditConfig = Field(default_factory=AuditConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description=(
            "Additional or replacement named provider profiles. Credential values "
            "are read from each profile's api_key_env at runtime."
        ),
    )
    perturbation_agent: PerturbationAgentConfig = Field(
        default_factory=PerturbationAgentConfig
    )
    planner_agent: PlannerAgentConfig = Field(default_factory=PlannerAgentConfig)
    summary_agent: SummaryAgentConfig = Field(default_factory=SummaryAgentConfig)
    validator_agent: ValidatorAgentConfig = Field(default_factory=ValidatorAgentConfig)

    @model_validator(mode="after")
    def _provider_references_exist(self) -> "AuditRunConfig":
        invalid_profile_names = sorted(
            name
            for name in self.providers
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name)
        )
        if invalid_profile_names:
            raise ValueError(
                "Invalid provider profile names: " + ", ".join(invalid_profile_names)
            )

        known = BUILTIN_PROVIDER_NAMES | set(self.providers)
        references = {
            "judge": self.judge.provider,
            "perturbation_agent": self.perturbation_agent.provider,
            "planner_agent": self.planner_agent.provider,
            "summary_agent": self.summary_agent.provider,
            "validator_agent": self.validator_agent.provider,
        }
        unknown = [
            f"{role}.provider={provider!r}"
            for role, provider in references.items()
            if provider is not None and provider not in known
        ]
        if unknown:
            raise ValueError("Unknown provider profile: " + ", ".join(unknown))
        return self

    def resolved_planner_provider(self) -> str:
        return self._resolved_role_provider(self.planner_agent)

    def resolved_planner_model(self) -> str:
        return self._resolved_role_model(self.planner_agent)

    def resolved_planner_decoding_params(self) -> dict[str, Any]:
        return self._resolved_role_decoding_params(self.planner_agent)

    def resolved_summary_model(self) -> str:
        return self._resolved_role_model(self.summary_agent)

    def resolved_summary_provider(self) -> str:
        return self._resolved_role_provider(self.summary_agent)

    def resolved_summary_decoding_params(self) -> dict[str, Any]:
        return self._resolved_role_decoding_params(self.summary_agent)

    def resolved_validator_model(self) -> str:
        return self._resolved_role_model(self.validator_agent)

    def resolved_validator_provider(self) -> str:
        return self._resolved_role_provider(self.validator_agent)

    def resolved_validator_decoding_params(self) -> dict[str, Any]:
        return self._resolved_role_decoding_params(self.validator_agent)

    def _resolved_role_provider(
        self,
        role: PlannerAgentConfig | SummaryAgentConfig | ValidatorAgentConfig,
    ) -> str:
        return role.provider or self.perturbation_agent.provider

    def _resolved_role_model(
        self,
        role: PlannerAgentConfig | SummaryAgentConfig | ValidatorAgentConfig,
    ) -> str:
        return role.model or self.perturbation_agent.model

    def _resolved_role_decoding_params(
        self,
        role: PlannerAgentConfig | SummaryAgentConfig | ValidatorAgentConfig,
    ) -> dict[str, Any]:
        return (
            role.decoding_params
            if role.decoding_params is not None
            else self.perturbation_agent.decoding_params
        )


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    input: str = Field(
        description="The complete agent input, including any task or system instructions."
    )
    output: Any = Field(
        description=(
            "The complete agent output being graded. Pairwise output contains "
            "`candidates` and `position_to_candidate`."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    gold_label: Any | None = None
    human_label_distribution: Any | None = None

    @model_validator(mode="after")
    def _validate_pairwise_identity(self) -> "Item":
        if not self.is_pairwise:
            return self
        if set(self.output) != {"candidates", "position_to_candidate"}:
            raise ValueError(
                "pairwise output must contain exactly candidates and position_to_candidate"
            )
        if not isinstance(self.output["candidates"], dict) or not isinstance(
            self.output["position_to_candidate"], dict
        ):
            raise ValueError("pairwise candidates and position_to_candidate must be mappings")
        if set(self.position_to_candidate) != {"A", "B"}:
            raise ValueError("pairwise position_to_candidate must contain exactly A and B")
        candidate_ids = list(self.position_to_candidate.values())
        if len(set(candidate_ids)) != 2:
            raise ValueError("pairwise A and B positions must map to different candidates")
        missing = set(candidate_ids) - set(self.candidates)
        if missing:
            raise ValueError(
                "position_to_candidate references missing candidate IDs: "
                + ", ".join(sorted(missing))
            )
        if self.gold_label is not None and str(self.gold_label) not in self.candidates:
            raise ValueError("pairwise gold_label must be a stable candidate ID")
        return self

    @property
    def is_pairwise(self) -> bool:
        return isinstance(self.output, dict) and (
            "candidates" in self.output or "position_to_candidate" in self.output
        )

    @property
    def candidates(self) -> dict[str, Any]:
        return self.output["candidates"] if self.is_pairwise else {}

    @property
    def position_to_candidate(self) -> dict[Literal["A", "B"], str]:
        return self.output["position_to_candidate"] if self.is_pairwise else {}

    def candidate_id_at(self, position: Literal["A", "B"]) -> str:
        return self.position_to_candidate[position]

    def candidate_at(self, position: Literal["A", "B"]) -> Any:
        return self.candidates[self.candidate_id_at(position)]


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    variant_id: str
    run_id: str
    raw_judge_output: str
    parsed_verdict: dict[str, Any] | str | int | float | bool | None
    call_status: Literal["ok", "transport_error"] = "ok"
    parse_status: Literal["ok", "error", "not_attempted"] = "ok"
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class Perturbation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_id: str
    item_id: str = "*"
    family: PerturbationFamily
    operator: str
    expected_effect: PerturbationExpectedEffect
    expected_relation: ExpectedRelation = ExpectedRelation.INVARIANT
    changed_fields: list[str]
    validity_status: Literal["accepted", "rejected", "unavailable", "not_needed"] = "not_needed"
    metadata: dict[str, Any] = Field(default_factory=dict)
    judge_config_overrides: dict[str, Any] = Field(default_factory=dict)
    item_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _relation_matches_effect(self) -> "Perturbation":
        if self.expected_effect == PerturbationExpectedEffect.WORSE_VERDICT:
            self.expected_relation = ExpectedRelation.DIRECTIONAL
        elif self.expected_effect == PerturbationExpectedEffect.SAME_CANDIDATE:
            self.expected_relation = ExpectedRelation.EQUIVARIANT
        return self


class ItemRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    original_verdict: Any
    original_score: float | None = None
    gold_label: Any | None = None
    noise_risk: float | None = None
    flip_risk: float | None = None
    all_attempt_flip_risk: float | None = None
    excess_over_noise: float | None = None
    invariant_attempts: int = 0
    invariant_analyzable: int = 0
    boundary_proximity: float = 0.0
    family_risks: dict[str, float] = Field(default_factory=dict)
    family_signed_shifts: dict[str, float] = Field(default_factory=dict)
    # Read legacy measurements without publishing them in new run artifacts.
    repeats_to_stable: int | None = Field(default=None, exclude=True)
    routing_rank: int | None = None
    judge_correct: bool | None = None
    judge_error: bool | None = None
    gold_error: bool | None = None
    directional_cases: int = 0
    directional_missed: int = 0
    directional_contradictions: int = 0
    directional_risk: float | None = None
    equivariant_cases: int = Field(default=0, exclude=True)
    equivariant_inconsistent: int = Field(default=0, exclude=True)
    equivariant_risk: float | None = Field(default=None, exclude=True)
