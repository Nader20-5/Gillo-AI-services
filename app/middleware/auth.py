from typing import Optional, List
from fastapi import Header, HTTPException, status, Depends
from jose import jwt, JWTError
from pydantic import BaseModel
from app.config import settings


class CurrentUser(BaseModel):
    user_id: str
    role: str
    project_id: Optional[str] = None
    all_projects: bool = False


async def get_current_user(
    authorization: Optional[str] = Header(None),
    x_mock_user_id: Optional[str] = Header(None),
    x_mock_role: Optional[str] = Header(None),
    x_mock_project_id: Optional[str] = Header(None),
    x_mock_all_projects: Optional[str] = Header(None),
) -> CurrentUser:
    """
    Extract caller identity from Bearer JWT.
    Supports mock development headers if Authorization header is absent.
    Enforces that Admin is blocked from AI operations.
    Enforces fail-closed scoping: non-admin callers must possess project_id or all_projects=True.
    """
    user: Optional[CurrentUser] = None

    # 1. Bearer JWT path
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": True, "require_exp": True}
            )
            user_id = payload.get("sub") or payload.get("user_id")
            role = payload.get("role", "Developer")
            project_id = payload.get("project_id")
            all_projects = bool(payload.get("all_projects", False))
            if not user_id:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token: missing subject")
            user = CurrentUser(user_id=user_id, role=role, project_id=project_id, all_projects=all_projects)
        except JWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired JWT token")

    # 2. Mock auth fallback (ONLY allowed when ALLOW_MOCK_AUTH is True and NOT production)
    elif settings.ALLOW_MOCK_AUTH and x_mock_user_id and x_mock_role:
        if settings.ENVIRONMENT.lower() == "production":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Mock authentication headers are strictly disabled in production."
            )
        all_projects = (x_mock_all_projects or "").lower() == "true"
        user = CurrentUser(
            user_id=x_mock_user_id,
            role=x_mock_role,
            project_id=x_mock_project_id,
            all_projects=all_projects
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header; mock authentication headers are disabled."
        )

    # Strict architectural guardrail: Admin role is forbidden from all AI endpoints
    if user.role.lower() == "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role is strictly forbidden from accessing AI endpoints."
        )

    # Scoping Fail-Closed: non-admin roles require project_id or explicit all_projects claim
    if not user.project_id and not user.all_projects:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Token must specify a 'project_id' or claim 'all_projects': true."
        )

    return user


def require_roles(allowed_roles: List[str]):
    """
    Enforces role-based access.
    Explicit guardrail: 'Admin' is strictly blocked from all AI endpoints.
    """
    async def role_checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role.lower() == "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role is strictly forbidden from accessing AI endpoints."
            )

        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: role '{user.role}' is not in allowed roles: {allowed_roles}"
            )
        return user

    return role_checker
