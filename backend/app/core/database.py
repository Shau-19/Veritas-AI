from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings


engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,        # tests connection liveness before every checkout
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,           # max wait to acquire a connection from pool (seconds)
    pool_recycle=1800,         # recycle connections after 30 min — prevents stale TCP
    connect_args={
        "server_settings": {"application_name": "veritas-ai-backend"},
        "command_timeout": 60,  # per-query timeout (seconds)
    },
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
