import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from app.config import settings

logger = logging.getLogger(__name__)


engine = create_async_engine(settings.database_url, echo=settings.debug)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


DEEPEVAL_COLUMNS = [
    "deepeval_faithfulness_score",
    "deepeval_hallucination_score",
    "deepeval_toxicity_score",
    "deepeval_bias_score",
    "deepeval_g_eval_score",
]
DEEPEVAL_COLUMN_DEFS = {
    "deepeval_faithfulness_score": "FLOAT",
    "deepeval_hallucination_score": "FLOAT",
    "deepeval_toxicity_score": "FLOAT",
    "deepeval_bias_score": "FLOAT",
    "deepeval_g_eval_score": "FLOAT",
}


async def init_db():
    for attempt in range(30):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

                for col in DEEPEVAL_COLUMNS:
                    try:
                        await conn.execute(
                            text(
                                f"ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS {col} {DEEPEVAL_COLUMN_DEFS[col]}"
                            )
                        )
                    except Exception:
                        pass

            logger.info("Database tables created successfully")
            return
        except Exception as e:
            if attempt < 29:
                logger.warning(f"Database not ready (attempt {attempt + 1}/30): {e}")
                await asyncio.sleep(4)
            else:
                logger.error(f"Database connection failed after 30 attempts: {e}")
                raise
