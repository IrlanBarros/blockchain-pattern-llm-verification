"""Constantes, taxonomias e defaults do pipeline."""

PIPELINE_VERSION = "0.6.1-gemini"
PROVIDER = "google-gemini"
API_FAMILY = "generateContent"

MANUAL_VERSION = "0.2"
CATALOG_VERSION = "v3"

# Modelos estáveis consultados na documentação oficial do Gemini API.
DEFAULT_STAGE1_MODEL = "gemini-3.1-flash-lite"
DEFAULT_STAGE2_MODEL = "gemini-3.5-flash"

# Gemini 3 é otimizado para temperature=1.0. A reprodutibilidade é reforçada
# por modelo estável, seed registrada, prompts/schemas congelados e auditoria.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SEED = 0
DEFAULT_STAGE1_THINKING_LEVEL = "minimal"
DEFAULT_STAGE2_THINKING_LEVEL = "low"
THINKING_LEVELS = ["minimal", "low", "medium", "high"]

DEFAULT_MAX_INPUT_CHARS = 12_000

# O Batch API inline aceita no máximo 20 MB por job. O pipeline também divide
# por bytes estimados, portanto este limite é apenas uma segunda barreira.
DEFAULT_BATCH_SIZE = 250
DEFAULT_BATCH_MAX_BYTES = 18_000_000
DEFAULT_POLL_SECONDS = 30

REQUIRED_INPUT_COLUMNS = {
    "repository",
    "issue_number",
    "issue_title",
    "issue_body",
}

OPTIONAL_INPUT_COLUMNS = {
    "repository_category": "",
    "concatenated_comments": "",
    "type": "",
    "labels": "",
    "state": "",
    "html_url": "",
}

REQUIRED_PATTERN_COLUMNS = {
    "pattern",
    "category",
    "subcategory",
    "description",
}

ISSUE_ACTIVITY_TYPES = [
    "feature_implementation",
    "bug_report",
    "bug_fix",
    "security",
    "refactoring",
    "performance_or_gas",
    "maintenance",
    "migration_or_upgrade",
    "testing_or_verification",
    "documentation",
    "build_or_dependency",
    "support_question",
    "conceptual_discussion",
    "other",
]

CHALLENGE_CATEGORIES = [
    "implementation_complexity",
    "security_risk",
    "gas_or_performance",
    "upgradeability",
    "migration",
    "interoperability",
    "data_consistency",
    "access_control",
    "testing_and_verification",
    "tooling_or_dependency",
    "documentation_or_understanding",
    "governance",
    "privacy_or_confidentiality",
    "reliability_or_availability",
    "other",
    "none_explicit",
]

ADOPTION_STATUSES = [
    "implemented_existing",
    "implementation_in_progress",
    "problem_with_implementation",
    "replacement_or_removal",
    "proposed_or_planned",
    "conceptual_discussion",
    "superficial_mention",
    "not_related",
    "insufficient_context",
]

VERDICTS = ["yes", "no", "uncertain", "insufficient_context"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]
FALSE_FRIEND_VALUES = ["yes", "no", "uncertain"]
EVIDENCE_LOCATIONS = ["title", "body", "comment", "pull_request_description"]
CONTEXT_STATUSES = ["sufficient", "insufficient_context"]

KNOWN_OVERLAP_GROUPS = [
    [
        "Proxy contract",
        "Contract Registry (on-chain)",
        "Off-chain Contract Registry",
        "Router contract",
        "Data contract",
    ],
    [
        "Ownable",
        "Role-based control",
        "Permissioned actions",
        "Limited-access",
        "Implicit authorization",
    ],
    [
        "Oracle",
        "Ticker tape",
        "Pull-based inbound oracle",
        "Push-based inbound oracle",
        "Reverse Oracle",
        "Data refresh",
    ],
    ["Off-chain data storage", "Digital Record", "State anchoring"],
    ["Mutex", "Check-Effects-Interactions", "Guard check"],
    ["Data compression", "Packing variables", "Packing booleans", "Unit vs Unit256"],
    ["Off-chain computation", "Delegated Computation"],
    ["Vote", "Poll"],
]
