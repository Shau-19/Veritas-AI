import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from app.api.routes.admin import get_admin_stats, get_current_admin
from app.models.user import User


# 1. Test get_current_admin dependency
async def test_get_current_admin_dependency():
    mock_db = AsyncMock()
    
    # User exists and is admin
    admin_user = User(id="user-admin", email="admin@veritas.ai", is_admin=True)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = admin_user
    mock_db.execute.return_value = mock_result
    
    res = await get_current_admin(user_id="user-admin", db=mock_db)
    assert res.id == "user-admin"
    assert res.is_admin is True

    # User exists but is NOT admin -> raises 403
    normal_user = User(id="user-normal", email="user@veritas.ai", is_admin=False)
    mock_result.scalar_one_or_none.return_value = normal_user
    
    try:
        await get_current_admin(user_id="user-normal", db=mock_db)
        assert False, "Expected 403 Forbidden"
    except HTTPException as e:
        assert e.status_code == 403
        assert "Admin privileges required" in e.detail


# 2. Test get_admin_stats route logic
async def test_get_admin_stats_route():
    admin_user = User(id="user-admin", email="admin@veritas.ai", is_admin=True)
    normal_user = User(id="user-normal", email="user@veritas.ai", is_admin=False)
    
    mock_db = AsyncMock()

    class MockResult:
        def __init__(self, val):
            self._val = val
        def scalar_one(self):
            return self._val
        def scalars(self):
            mock_scalars = MagicMock()
            mock_scalars.all.return_value = self._val if isinstance(self._val, list) else [self._val]
            return mock_scalars

    # Setup database mocks for count queries and list queries
    async def mock_execute(query):
        query_str = str(query).lower()
        if "count" in query_str:
            if "users" in query_str:
                return MockResult(2)
            elif "datasets" in query_str:
                return MockResult(5)
        elif "users" in query_str:
            return MockResult([admin_user, normal_user])
        elif "training_jobs" in query_str:
            return MockResult([]) # empty jobs list
        return MockResult(None)

    mock_db.execute = mock_execute

    # Call route
    stats = await get_admin_stats(admin=admin_user, db=mock_db)
    assert "summary" in stats
    assert "users" in stats
    assert stats["summary"]["total_users"] == 2
    assert stats["summary"]["total_datasets"] == 5
    assert len(stats["users"]) == 2
    assert stats["users"][0]["email"] == "admin@veritas.ai"
    assert stats["users"][1]["email"] == "user@veritas.ai"

    print("[OK] test_admin_endpoints passed")


if __name__ == "__main__":
    print("=== Running Admin Access Dashboard Routing Tests ===")
    asyncio.run(test_get_current_admin_dependency())
    asyncio.run(test_get_admin_stats_route())
    print("=== Admin endpoint tests passed successfully! ===")
