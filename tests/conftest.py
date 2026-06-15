import sys
from unittest.mock import patch, AsyncMock, MagicMock

import pytest_asyncio
import uuid
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker,
)

_mock_st = MagicMock()
_mock_st.SentenceTransformer = MagicMock
_mock_st.util = MagicMock()
_mock_st.util.cos_sim = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.5)))
sys.modules["sentence_transformers"] = _mock_st

from app.database import Base
from app.main import app
from app.models.prompt import Prompt
from app.models.llm_model import LLMModel
from app.models.alert import Alert, AlertMetric, AlertOperator, AlertStatus

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_session_factory = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session(setup_db):
    async with test_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_session):
    from app.database import get_db

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with patch("app.main.init_db", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def sample_prompt(db_session):
    prompt = Prompt(
        id=uuid.uuid4(),
        name="test_prompt",
        content="What is the capital of France?",
        category="qa",
        expected_output="Paris",
        tags="test,geography",
    )
    db_session.add(prompt)
    await db_session.commit()
    await db_session.refresh(prompt)
    return prompt


@pytest_asyncio.fixture
async def sample_model(db_session):
    model = LLMModel(
        id=uuid.uuid4(),
        name="test-model",
        provider="ollama",
        model_type="open-source",
        context_window=4096,
        is_active=True,
        gpu_required=False,
        cost_per_prompt_token=0.000003,
        cost_per_completion_token=0.000015,
    )
    db_session.add(model)
    await db_session.commit()
    await db_session.refresh(model)
    return model


@pytest_asyncio.fixture
async def sample_alert(db_session):
    alert = Alert(
        id=uuid.uuid4(),
        name="test-alert",
        metric=AlertMetric.latency,
        operator=AlertOperator.gt,
        threshold=5000.0,
        status=AlertStatus.active,
    )
    db_session.add(alert)
    await db_session.commit()
    await db_session.refresh(alert)
    return alert
