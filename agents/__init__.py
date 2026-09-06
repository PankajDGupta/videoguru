"""VideoGuru Agents Package."""

from agents.agent import app, get_app, get_root_agent, root_agent
from agents.curation import (
    CURATION_INSTRUCTION,
    CurationAgent,
    create_curation_agent,
    curation_agent,
)
from agents.ingestion import (
    INGESTION_INSTRUCTION,
    IngestionAgent,
    StubIngestionAgent,
    create_ingestion_agent,
    ingestion_agent,
)
from agents.root_greeter import (
    ROOT_GREETER_INSTRUCTION,
    RootGreeterAgent,
    create_root_greeter_agent,
    root_greeter_agent,
)

__all__ = [
    "CURATION_INSTRUCTION",
    "CurationAgent",
    "INGESTION_INSTRUCTION",
    "IngestionAgent",
    "ROOT_GREETER_INSTRUCTION",
    "RootGreeterAgent",
    "StubIngestionAgent",
    "app",
    "create_curation_agent",
    "create_ingestion_agent",
    "create_root_greeter_agent",
    "curation_agent",
    "get_app",
    "get_root_agent",
    "ingestion_agent",
    "root_agent",
    "root_greeter_agent",
]

