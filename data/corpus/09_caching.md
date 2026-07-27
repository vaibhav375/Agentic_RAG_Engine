# Response caching

Breeze can cache handler responses with the `@cache(ttl=...)` decorator. The
`ttl` argument is the time to live in seconds; after it elapses the cached entry
expires and the handler runs again.

## Cache keys

By default the cache key is derived from the request path and the sorted query
string, so `/items?a=1&b=2` and `/items?b=2&a=1` map to the same key. Request
headers and the request body are not part of the key. To include a header, pass
`vary=["Authorization"]` to the decorator.

## Only GET is cached

Only `GET` and `HEAD` requests are cached. Requests using `POST`, `PUT`, `PATCH`,
or `DELETE` are never cached, because they are expected to change server state.

## Manual invalidation

Call `cache.invalidate(path)` to drop all cached entries for a path immediately,
regardless of their remaining TTL. This is useful after a write that makes cached
reads stale.
