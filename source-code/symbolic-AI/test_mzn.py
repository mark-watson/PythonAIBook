# test_mzn.py - Simple MiniZinc constraint satisfaction from Python
#
# Solves for integer variables x and y given two constraints:
#   x + y = n   (sum constraint)
#   x * y = m   (product constraint)
#
# The MiniZinc model (test_mzn.mzn) declares x and y as decision variables
# with domain 1..n. The Python script sets the constants n and m, invokes
# the solver, and prints the solution.
#
# Requirements: brew install minizinc && uv pip install minizinc

from minizinc import Instance, Model, Solver

# Use the CBC (Coin-or Branch and Cut) solver installed with minizinc
coinbc = Solver.lookup("coinbc")

# Load the MiniZinc model and create a solver instance
test1 = Model("./test_mzn.mzn")
instance = Instance(coinbc, test1)

# Set the problem parameters: find x, y where x+y=30 and x*y=200
instance["n"] = 30
instance["m"] = 200

# Solve and print the result (expected: x=20, y=10)
result = instance.solve()
print(result)
print(result["x"])
print(result["y"])
