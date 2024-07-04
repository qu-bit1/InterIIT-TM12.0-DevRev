planning_template = """
Given a query and a list of APIs with their arguments, find whether the APIs can solve the query or not. If not, return empty list. Else, extract values or variables from the query that correspond to each argument of the tools that needs to be called. If some argument value isn't known, use tools that can retrieve those argument values. Return a JSON file containing the tools in the order they need to be called. Identify cases where an API argument depends on the output of a previous API call. Replace such arguments with the notation $$PREV[i] where 'i' represents the order of the previous API call in the chain. Do NOT explain yourself, only return the JSON.

Query: {query}
Tools: {tools}

Answer:
"""

improvement_template = """
Given a list of examples, each containing a query and the corresponding APIs to be called to solve the query, find whether the JSON present in current solution solves the current query.
If yes, modify its format to match with that of solutions in examples and return the modified JSON.
If not, modify the current JSON solution to include necessary tools from list of available tools.
If no solution is available, return a blank list.
Remember to use double quotes instead of single quotes in JSON output. Do NOT explain yourself, only return the JSON.

Examples : {examples}

Current Query : {query}
Current Solution : {solution}

Answer:
"""

optimization_template = """
Given a query, solution and available APIs, optimize the number of APIs calls in the solution. Do not use API calls unless absolutely necessary. Look for redundancy and check for mistakes in the combination of API calls. Return current solution if no changes are necessary. Do NOT explain yourself, only return the JSON.

Current Query : {query}
Current Solution : {solution}
Available APIs : {tools}

Final solution (same format as current solution):
"""
