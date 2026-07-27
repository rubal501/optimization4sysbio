Implementation of *Evolutionary Algorithm for Complex-process Optimization* (Egea, Martí & Banga, 2009), compatible with SciMLBase.



## How ot use

```julia
include("EACOP.jl")
using .EACOPOptimizer. SciMLBase, Optim

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
cons(res, x, p) = (res[1] = x[1]^2 + x[2]^2)

f = OptimizationFunction(rosenbrock; cons = cons)
prob = OptimizationProblem(f, zeros(2), [1.0, 100.0];
                                  lb = [-5.0, -5.0], ub = [5.0, 5.0],
                                  lcons = [-Inf], ucons = [1.0])

sol = solve(prob, EACOP(penalty = 1e5); maxiters = 300)

sol.u
```

## TODO:
