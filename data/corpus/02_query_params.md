# Query parameters

Any handler argument that is not part of the path is interpreted as a query
parameter. For example, in `def list_items(skip: int = 0, limit: int = 10)` both
`skip` and `limit` are read from the query string, so a request to
`/items?skip=20&limit=50` sets `skip` to 20 and `limit` to 50.

## Defaults and required parameters

A query parameter with a default value is optional. A query parameter without a
default value is required, and Breeze returns a 422 response if it is missing.

## Boolean parsing

Boolean query parameters accept the values `1`, `true`, `on`, and `yes` as True,
and `0`, `false`, `off`, and `no` as False. Parsing is case-insensitive, so
`True` and `TRUE` are both accepted.

## Lists

To receive multiple values for the same query key, annotate the argument as a
list, for example `tags: list[str]`. A request to `/items?tags=a&tags=b` then
sets `tags` to `["a", "b"]`. If no values are provided, the list defaults to
empty.
