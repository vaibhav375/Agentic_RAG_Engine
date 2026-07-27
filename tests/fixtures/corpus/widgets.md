# Widgets

A widget is created by calling `make_widget(name)`. The name must be unique.

## Colors

The default widget color is blue. To change it, pass the `color` argument, for
example `make_widget("w1", color="red")`.

## Limits

A single account may create at most 50 widgets. Attempting to create a 51st
widget returns an error with status code 429.
