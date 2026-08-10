# Routing

Breeze maps incoming HTTP requests to handler functions using route decorators.
You register a route by decorating a function with an HTTP-method decorator such
as `@app.get`, `@app.post`, `@app.put`, `@app.patch`, or `@app.delete`.

## Path parameters

A path parameter is declared by placing the parameter name in curly braces inside
the path, for example `@app.get("/items/{item_id}")`. The value is passed to the
handler as a function argument with the same name. By default a path parameter is
a string. To convert it to another type, add a type annotation to the argument;
for example `def read_item(item_id: int)` will parse the value as an integer and
return a 422 response if the value cannot be converted.

## Path ordering

Routes are matched in the order they are declared. A fixed path such as
`/users/me` must be declared before a parameterized path such as
`/users/{user_id}`; otherwise the fixed path is never reached because the
parameterized route matches first.

## Route names and URL building

Every route may be given a name via the `name` argument, for example
`@app.get("/items/{item_id}", name="item_detail")`. You can build a URL for a
named route with `app.url_for("item_detail", item_id=42)`, which returns the
string `/items/42`.
