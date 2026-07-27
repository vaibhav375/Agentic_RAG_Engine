# Configuration

Application settings are defined by subclassing `breeze.Settings` and declaring
typed fields. Values are read from environment variables whose names match the
field names in uppercase.

```python
class Settings(breeze.Settings):
    debug: bool = False
    database_url: str
    workers: int = 4
```

At startup Breeze reads `DEBUG`, `DATABASE_URL`, and `WORKERS` from the
environment. A field without a default is required; if `DATABASE_URL` is unset,
the application fails to start with a clear error naming the missing variable.

## .env files

If a file named `.env` exists in the working directory, Breeze loads it before
reading settings, so values there populate the environment. Real environment
variables always take precedence over values in the `.env` file.

## Debug mode

When `debug` is `True`, Breeze includes the exception traceback in 500 responses
and enables automatic reloading on file changes. Debug mode must never be enabled
in production because it exposes internal details.

## Accessing settings

Settings are available through `app.settings` and can be injected into any
handler with `Depends(get_settings)`. The settings object is created once at
startup and reused for every request.
