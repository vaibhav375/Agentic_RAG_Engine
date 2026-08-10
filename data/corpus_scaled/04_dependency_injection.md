# Dependency injection

Breeze provides a dependency-injection system through the `Depends` marker. A
dependency is any callable that returns a value; you request it by giving a
handler argument the default `Depends(your_callable)`.

## Declaring a dependency

```python
def get_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()

@app.get("/items")
def list_items(db = Depends(get_db)):
    return db.all("items")
```

Because `get_db` uses `yield`, the code after `yield` runs as teardown after the
response is sent. This is how Breeze closes database connections and releases
resources per request.

## Caching within a request

By default a dependency is called at most once per request. If two different
dependencies both depend on `get_db`, `get_db` still runs only a single time and
the same value is shared. You can disable this by passing
`Depends(get_db, use_cache=False)`, which forces the dependency to run every time
it is requested.

## Sub-dependencies

A dependency may itself declare dependencies, forming a tree. Breeze resolves the
tree depth-first and injects the results, so a dependency that needs the current
user can depend on another dependency that reads the auth token.
