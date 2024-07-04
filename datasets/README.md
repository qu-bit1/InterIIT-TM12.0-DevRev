# Dataset

## Overview

This repository contains a dataset comprising a comprehensive list of tools, associated queries, and examples utilized in various contexts.

## Contents

### Tools

Format:

```json
{
    "tool_name":  ,
    "description":  ,
    "arguments":[
            {
                "argument_name": ,
                "argument_description": ,
                "argument_type": ,
                "examples":[]
            }
          ]
}
```

- [tools.json](tools.json) : A compilation of tools along with their description and arguments. Directly taken from the problem statement.
- [tools_augmented](tools_augmented.json) : An amalgation of tools sourced from DevRev API and Jira Open API.
- [tools_complete](tools_complete.json) : A compilation of all the available tools.

### Queries

Format:

```json
[
    "queries"
]
```

- [queries](queries.json) : Comprises of a set of distinct queries, directly taken from the problem statement.
- [queries_augmented](queries_augmented.json) : Comprises of a set of queries based on tools obtained from DevRev API and Jira Open API.

### Examples

Format:

```json
    {
        "Query": ,
        "Solution": [
            {
                "tool_name": ,
                "arguments": [
                    {
                        "argument_name": ,
                        "argument_value": 
                    }
                ]
            }
        ]
    }
```

- [examples](examples.json) : A collection of examples showcasing tool-query relationships, directly obtained from the probem statement.
- [examples_augmented](examples_augmented.json) : A collection of examples showcasing tool-query relationships, based on the list of augmented tools.

### Naming Convention

[tools.json](tools.json), [examples.json](examples.json) and [queries.json](queries.json) form the DevRev (small) dataset.

[tools_augmented.json](tools_augmented.json), [examples_augmented.json](examples_augmented.json) and [queries_augmented.json](queries_augmented.json) form the DevRev (large) dataset.
