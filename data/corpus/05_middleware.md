# Middleware

Middleware is code that runs for every request before it reaches a handler and
for every response before it is returned to the client. You add middleware with
`app.add_middleware(MiddlewareClass, **options)`.

## Execution order

Middleware runs in the reverse of the order it is added on the way in, and in the
order it is added on the way out. In other words, the middleware added last wraps
the handler most tightly. Plan the order carefully when one middleware depends on
another.

## Built-in middleware

Breeze ships with three middleware classes. `CORSMiddleware` adds cross-origin
headers. `GZipMiddleware` compresses responses larger than a configurable
`minimum_size`, which defaults to 500 bytes. `TrustedHostMiddleware` rejects
requests whose `Host` header is not in an allowed list, returning a 400 response.

## Writing custom middleware

A custom middleware is a class with an async `dispatch(self, request, call_next)`
method. Call `response = await call_next(request)` to pass control down the chain,
then modify and return the response. Raising an exception in `dispatch` short-
circuits the request and skips the handler entirely.
