#!/usr/bin/env bash

jq --indent 1 \
    '
    (.cells[] | select(has("outputs")) | .outputs) = []
    | (.cells[] | select(has("execution_count")) | .execution_count) = null
    | .metadata = {"kernelspec": {"display_name": "firedrake", "language": "python", "name": "firedrake"},
                   "language_info": {"name":"python", "pygments_lexer": "ipython3"}}
    | .cells[].metadata = {}
    ' $1 | sponge $1

