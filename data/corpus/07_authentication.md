# Authentication

Breeze does not enforce any authentication scheme by default; every route is
public until you add a security dependency. The recommended pattern is to write a
dependency that validates credentials and inject it into protected routes.

## API key dependency

```python
def require_api_key(x_api_key: str = Header()):
    if x_api_key not in VALID_KEYS:
        raise HTTPError(401, "invalid api key")
    return x_api_key
```

Adding `Depends(require_api_key)` to a route makes the key mandatory. When the
key is missing or wrong, the dependency raises `HTTPError(401)` and the handler
never runs.

## Bearer tokens

For token auth, read the `Authorization` header, which has the form
`Bearer <token>`. Breeze provides `BearerAuth` as a dependency that extracts the
token and returns it, raising a 401 if the header is absent or malformed.

## Scopes

A token may carry scopes. The `require_scope("admin")` dependency checks that the
authenticated token includes the named scope and raises a 403 Forbidden response
if it does not. Note that 401 means "not authenticated" while 403 means
"authenticated but not allowed".
