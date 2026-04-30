# us_states.py - Four-color map coloring of US states using MiniZinc
#
# Solves the classic map coloring problem: assign one of 4 colors to each
# US state such that no two neighboring states share the same color.
# This is an application of the Four Color Theorem.
#
# The MiniZinc model (us_states.mzn) defines adjacency constraints for
# all 50 states. The Python script sets the number of allowed colors,
# invokes the constraint solver, and prints each state's assigned color.
#
# Requirements: brew install minizinc && uv pip install minizinc

from minizinc import Instance, Model, Solver

# Use the CBC (Coin-or Branch and Cut) solver installed with minizinc
coinbc = Solver.lookup("coinbc")

# Load the US states adjacency model
model = Model("./us_states.mzn")
instance = Instance(coinbc, model)
instance["nc"] = 4  # solve for a maximum of 4 colors (3 is unsatisfiable)

# Solve the constraint satisfaction problem
result = instance.solve()
print(result)

# Extract and display each state's assigned color number
all_states = list(result.__dict__['solution'].__dict__.keys())
all_states.remove('_checker')  # internal MiniZinc field, not a state
print(all_states)
for state in all_states:
    print(f" {state} \t: \t{result[state]}")
