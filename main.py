# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gurobipy as gp



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Create a new model
    m = gp.Model()

    # Create variables
    x = m.addVar(vtype='B', name="x")
    y = m.addVar(vtype='B', name="y")
    z = m.addVar(vtype='B', name="z")

    # Set objective function
    m.setObjective(x + y + 2 * z, gp.GRB.MAXIMIZE)

    # Add constraints
    m.addConstr(x + 2 * y + 3 * z <= 4)
    m.addConstr(x + y >= 1)

    # Solve it!
    m.optimize()

    print(f"Optimal objective value: {m.objVal}")
    print(f"Solution values: x={x.X}, y={y.X}, z={z.X}")