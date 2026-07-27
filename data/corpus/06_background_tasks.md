# Background tasks

A background task is work that runs after the response has been sent, so the
client does not wait for it. Request a `BackgroundTasks` object by adding an
argument of that type to your handler, then call `tasks.add_task(fn, *args)`.

```python
@app.post("/send")
def send(email: str, tasks: BackgroundTasks):
    tasks.add_task(write_log, email)
    return {"status": "queued"}
```

## Execution guarantees

Background tasks run in the same process as the server, after the response is
flushed. They are executed in the order they were added. Because they share the
process, a crash in the server process loses any pending tasks; Breeze does not
persist them. For durable work, use an external queue instead.

## Error handling

An exception raised inside a background task does not affect the response, which
has already been sent. The exception is logged through the application logger at
`ERROR` level and the remaining tasks still run.
