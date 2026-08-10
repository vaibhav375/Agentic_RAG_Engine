# Request bodies

To receive a JSON request body, declare a handler argument whose type is a
`Schema` subclass. Breeze reads the request body, validates it against the
schema, and passes an instance to the handler.

## Defining a schema

A schema is a class that inherits from `breeze.Schema` and declares typed
fields, for example:

```python
class Item(breeze.Schema):
    name: str
    price: float
    in_stock: bool = True
```

Fields without a default are required. The `in_stock` field above is optional and
defaults to `True`.

## Validation errors

If the incoming body is missing a required field or a field has the wrong type,
Breeze returns a 422 Unprocessable Entity response whose body lists each invalid
field and the reason. Validation runs before your handler is called, so the
handler only ever sees valid data.

## Combining body, path, and query

A single handler may declare a path parameter, one or more query parameters, and
a body schema at the same time. Breeze resolves each argument by its source: path
parameters from the URL path, schema arguments from the JSON body, and everything
else from the query string.
