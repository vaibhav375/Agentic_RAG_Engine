# Error handling

To return an HTTP error from a handler or dependency, raise `HTTPError(status,
detail)`. Breeze converts it into a JSON response of the form
`{"detail": "<detail>"}` with the given status code.

## Custom exception handlers

Register a handler for a specific exception type with
`@app.exception_handler(MyError)`. The decorated function receives the request
and the exception and must return a response. This lets you map domain
exceptions to consistent HTTP responses in one place.

## Validation error shape

Validation failures produce a 422 response whose `detail` is a list of objects,
each with a `loc` (the location of the bad field), a `msg` (a human-readable
message), and a `type` (a machine-readable error code). This shape is stable and
safe to parse on the client.

## Default 500 behavior

If a handler raises an exception that is not an `HTTPError` and has no registered
handler, Breeze returns a generic 500 response with the body
`{"detail": "Internal Server Error"}` and logs the full traceback. The traceback
is never included in the response body, to avoid leaking internal details.
