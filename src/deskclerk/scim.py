# SCIM 2.0 (RFC 7643 and 7644) User endpoints under /scim/v2, so an identity provider can tell
# deskclerk that an employee left. Deactivating or deleting a user erases their data. The users
# the provider created are kept in one Postgres table; groups are not supported. The wire models,
# filter parser, PATCH engine and error bodies come from scim2-models.

import hmac
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from psycopg.errors import UniqueViolation
from pydantic import ValidationError
from scim2_models import (
    AuthenticationScheme,
    BaseModel,
    Bulk,
    ChangePassword,
    Context,
    Error,
    ETag,
    Filter,
    InvalidFilterException,
    InvalidSyntaxException,
    InvalidValueException,
    ListResponse,
    Meta,
    MutabilityException,
    NotFoundException,
    Patch,
    PatchOp,
    ResourceType,
    Schema,
    SCIMException,
    ScimProvider,
    SearchRequest,
    ServiceProviderConfig,
    Sort,
    UnauthorizedException,
    UniquenessException,
    User,
)
from scim2_models.path.expressions import CompareOperator, Comparison

from deskclerk.database import Database

MAX_PAGE = 100
UserModel = User[Any]
LOOKUP_ATTRIBUTES = {"username": "userName", "externalid": "externalId"}
CONFIG = ServiceProviderConfig(
    patch=Patch(supported=True),
    bulk=Bulk(supported=False, max_operations=0, max_payload_size=0),
    filter=Filter(supported=True, max_results=MAX_PAGE),
    change_password=ChangePassword(supported=False),
    sort=Sort(supported=False),
    etag=ETag(supported=False),
    authentication_schemes=[
        AuthenticationScheme(
            type=AuthenticationScheme.Type.oauthbearertoken,
            name="Bearer token",
            description="The SCIM token from DESKCLERK_SCIM_TOKEN",
            primary=True,
        )
    ],
)
PROVIDER = ScimProvider(models=[User], config=CONFIG)


class ScimResponse(JSONResponse):
    media_type = "application/scim+json"


@dataclass(frozen=True)
class ScimUser:
    id: str
    external_id: str
    user_name: str
    active: bool


class ScimUsers:
    def __init__(self, database: Database) -> None:
        self._pool = database.pool
        database.create_tables(
            "CREATE TABLE IF NOT EXISTS scim_users ("
            "position BIGSERIAL, id TEXT PRIMARY KEY, external_id TEXT NOT NULL UNIQUE, "
            "user_name TEXT NOT NULL, active BOOLEAN NOT NULL)",
            "CREATE UNIQUE INDEX IF NOT EXISTS scim_users_user_name "
            "ON scim_users (lower(user_name))",
        )

    def add(self, external_id: str, user_name: str, active: bool) -> ScimUser:
        user = ScimUser(str(uuid.uuid4()), external_id, user_name, active)
        with self._pool.connection() as connection:
            connection.execute(
                "INSERT INTO scim_users (id, external_id, user_name, active) "
                "VALUES (%s, %s, %s, %s)",
                (user.id, user.external_id, user.user_name, user.active),
            )
        return user

    def get(self, user_id: str) -> ScimUser | None:
        users = self._select("WHERE id = %s", (user_id,))
        return users[0] if users else None

    def find(self, attribute: str | None, value: str | None) -> list[ScimUser]:
        if attribute == "externalId":
            return self._select("WHERE external_id = %s", (value,))
        if attribute == "userName":
            return self._select("WHERE lower(user_name) = lower(%s)", (value,))
        return self._select("", ())

    def is_deactivated(self, external_id: str) -> bool:
        return any(not user.active for user in self.find("externalId", external_id))

    def save(self, user: ScimUser) -> None:
        with self._pool.connection() as connection:
            connection.execute(
                "UPDATE scim_users SET external_id = %s, user_name = %s, active = %s WHERE id = %s",
                (user.external_id, user.user_name, user.active, user.id),
            )

    def delete(self, user_id: str) -> None:
        with self._pool.connection() as connection:
            connection.execute("DELETE FROM scim_users WHERE id = %s", (user_id,))

    def _select(self, where: str, arguments: tuple[str | None, ...]) -> list[ScimUser]:
        with self._pool.connection() as connection:
            rows = connection.execute(
                "SELECT id, external_id, user_name, active FROM scim_users "  # noqa: S608
                f"{where} ORDER BY position",
                arguments,
            ).fetchall()
        return [ScimUser(row[0], row[1], row[2], row[3]) for row in rows]


@contextmanager
def _as_scim_errors() -> Iterator[None]:
    try:
        yield
    except ValidationError as invalid:
        raise SCIMException.from_error(Error.from_validation_errors(invalid)[0]) from None


def _lookup(search: SearchRequest[UserModel]) -> tuple[str | None, str | None]:
    if search.filter is None:
        return None, None
    node = search.filter.ast
    if (
        isinstance(node, Comparison)
        and node.op == CompareOperator.eq
        and node.attr_path.sub_attr is None
        and node.attr_path.uri is None
        and node.attr_path.attr.lower() in LOOKUP_ATTRIBUTES
        and isinstance(node.value, str)
    ):
        return LOOKUP_ATTRIBUTES[node.attr_path.attr.lower()], node.value
    raise InvalidFilterException(detail='only userName eq "..." and externalId eq "..."')


def _submitted(body: Any, context: Context) -> UserModel:
    with _as_scim_errors():
        submitted: UserModel = User.model_validate(body, scim_ctx=context)
    if submitted.user_name is None:
        raise InvalidValueException(detail="userName is required")
    return submitted


def _wire(model: BaseModel, context: Context) -> dict[str, Any]:
    return model.model_dump(scim_ctx=context, mode="json")


def mount_scim(
    app: FastAPI,
    token: str,
    public_url: str,
    users: ScimUsers,
    erase: Callable[[str], object],
) -> None:
    expected = token.encode()

    def require_token(authorization: str | None = Header(default=None)) -> None:
        scheme, _, presented = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(presented.encode(), expected):
            raise UnauthorizedException(detail="send Authorization: Bearer <SCIM token>")

    async def json_body(request: Request) -> Any:
        try:
            return await request.json()
        except ValueError:
            raise InvalidSyntaxException(detail="the body is not JSON") from None

    def location(user: ScimUser) -> str:
        return f"{public_url}/scim/v2/Users/{user.id}"

    def resource(user: ScimUser) -> UserModel:
        return User(
            id=user.id,
            external_id=user.external_id,
            user_name=user.user_name,
            active=user.active,
            meta=Meta(resource_type="User", location=location(user)),
        )

    def existing(user_id: str) -> ScimUser:
        user = users.get(user_id)
        if user is None:
            raise NotFoundException(detail=f"no user {user_id}")
        return user

    def save(user: ScimUser) -> ScimUser:
        try:
            users.save(user)
        except UniqueViolation:
            raise UniquenessException(detail="userName or externalId is already used") from None
        if not user.active:
            erase(user.external_id)
        return user

    router = APIRouter(
        prefix="/scim/v2",
        dependencies=[Depends(require_token)],
        default_response_class=ScimResponse,
    )

    @router.get("/ServiceProviderConfig")
    def service_provider_config() -> dict[str, Any]:
        return _wire(CONFIG, Context.RESOURCE_QUERY_RESPONSE)

    @router.get("/ResourceTypes")
    def resource_types() -> dict[str, Any]:
        found = list(PROVIDER.resource_types)
        listing = ListResponse[ResourceType](
            total_results=len(found), start_index=1, items_per_page=len(found), resources=found
        )
        return _wire(listing, Context.RESOURCE_QUERY_RESPONSE)

    @router.get("/Schemas")
    def schemas() -> dict[str, Any]:
        found = list(PROVIDER.schemas)
        listing = ListResponse[Schema](
            total_results=len(found), start_index=1, items_per_page=len(found), resources=found
        )
        return _wire(listing, Context.RESOURCE_QUERY_RESPONSE)

    @router.get("/Users")
    def list_users(request: Request) -> dict[str, Any]:
        wanted = {
            name: request.query_params[name]
            for name in ("filter", "startIndex", "count")
            if name in request.query_params
        }
        with _as_scim_errors():
            search = SearchRequest[UserModel].model_validate(
                wanted, scim_ctx=Context.SEARCH_REQUEST
            )
        matches = users.find(*_lookup(search))
        size = MAX_PAGE if search.count is None else min(search.count, MAX_PAGE)
        start = search.start_index_0 or 0
        page = [resource(user) for user in matches[start : start + size]]
        listing = ListResponse[UserModel](
            total_results=len(matches),
            start_index=start + 1,
            items_per_page=len(page),
            resources=page,
        )
        return _wire(listing, Context.RESOURCE_QUERY_RESPONSE)

    @router.post("/Users")
    def create_user(body: Any = Depends(json_body)) -> ScimResponse:
        new = _submitted(body, Context.RESOURCE_CREATION_REQUEST)
        if new.external_id is None or new.user_name is None:
            raise InvalidValueException(detail="externalId is required")
        try:
            user = users.add(new.external_id, new.user_name, new.active is not False)
        except UniqueViolation:
            raise UniquenessException(detail="userName or externalId is already used") from None
        if not user.active:
            erase(user.external_id)
        created = _wire(resource(user), Context.RESOURCE_CREATION_RESPONSE)
        return ScimResponse(created, 201, {"Location": location(user)})

    @router.get("/Users/{user_id}")
    def get_user(user_id: str) -> dict[str, Any]:
        return _wire(resource(existing(user_id)), Context.RESOURCE_QUERY_RESPONSE)

    @router.put("/Users/{user_id}")
    def replace_user(user_id: str, body: Any = Depends(json_body)) -> dict[str, Any]:
        user = existing(user_id)
        new = _submitted(body, Context.RESOURCE_REPLACEMENT_REQUEST)
        if new.external_id is not None and new.external_id != user.external_id:
            raise MutabilityException(detail="externalId cannot change")
        saved = save(replace(user, user_name=new.user_name or "", active=new.active is not False))
        return _wire(resource(saved), Context.RESOURCE_REPLACEMENT_RESPONSE)

    @router.patch("/Users/{user_id}")
    def patch_user(user_id: str, body: Any = Depends(json_body)) -> dict[str, Any]:
        user = existing(user_id)
        patched = resource(user)
        with _as_scim_errors():
            operations = PatchOp[UserModel].model_validate(
                body, scim_ctx=Context.RESOURCE_PATCH_REQUEST
            )
            operations.patch(patched)
        if patched.external_id != user.external_id:
            raise MutabilityException(detail="externalId cannot change")
        if not patched.user_name:
            raise InvalidValueException(detail="userName cannot be empty")
        if patched.active is None:
            raise MutabilityException(detail="active cannot be removed")
        saved = save(replace(user, user_name=patched.user_name, active=patched.active))
        return _wire(resource(saved), Context.RESOURCE_PATCH_RESPONSE)

    @router.delete("/Users/{user_id}", status_code=204)
    def delete_user(user_id: str) -> Response:
        user = existing(user_id)
        erase(user.external_id)
        users.delete(user_id)
        return Response(status_code=204)

    def refused(request: Request, raised: Exception) -> ScimResponse:
        error = raised if isinstance(raised, SCIMException) else SCIMException()
        headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else None
        return ScimResponse(
            _wire(error.to_error(), Context.RESOURCE_QUERY_RESPONSE), error.status, headers
        )

    app.add_exception_handler(SCIMException, refused)
    app.include_router(router)
