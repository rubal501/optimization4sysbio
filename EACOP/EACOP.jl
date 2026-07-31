"""
    EACOP.jl

Simple implementation of the *Evolutionary Algorithm for Complex-process Optimization*
(Egea, Martí & Banga, 2009), compatible with the `SciMLBase` optimization interface.

Implemented components from the article:
  * Initial population via Latin Hypercube Sampling (half by quality, half random).
  * Combination via wide and biased hyper-rectangles (Eqs. 9–14).
  * (1+1) selection/update: an offspring only enters the population by replacing its own parent.
  * *Go-beyond* strategy (Algorithm 1).
  * Local optima escape mechanism using the `nstuck` counter and `nchange` parameter.
  * Constraints handled via the static penalty term from Eq. 17 
    (L-∞ norm of constraint violations).

Additionally, `prob.sense` is respected: when `MaxSense` is specified, `-C(x)` is minimized internally.

Does not include the local search phase, as discussed in Section 1 of the article.
"""
module EACOPOptimizer

import SciMLBase
using QuasiMonteCarlo
using Random

export EACOP

# `Infeasible` does not exist in older versions of SciMLBase
const INFEASIBLE = isdefined(SciMLBase.ReturnCode, :Infeasible) ?
                   SciMLBase.ReturnCode.Infeasible : SciMLBase.ReturnCode.Failure

"""
    EACOPCache(prob)

Minimal cache that satisfies the `SciMLBase.AbstractOptimizationCache` interface.
`build_solution` only accepts a cache as its first argument (not an 
`OptimizationProblem`), and accesses `cache.p` and `cache.f` for symbolic 
indexing and observables. The original problem is also stored for convenience.
"""
struct EACOPCache{F, P, PR} <: SciMLBase.AbstractOptimizationCache
    f::F
    p::P
    prob::PR
end

EACOPCache(prob) = EACOPCache(prob.f, prob.p, prob)

# ---------------------------------------------------------------------------
# 1. Algorithm Definition and Hyperparameters
# ---------------------------------------------------------------------------

"""
    EACOP(; pop_multiplier = 10, nchange = 22, penalty = 1e6, rng = Random.default_rng())

Evolutionary Algorithm for Complex-process Optimization solver.

## Keyword Arguments
- `pop_multiplier`: Approximate number of new solutions generated per iteration, 
  per decision variable (`10 * nvar` in the article). Also determines the size 
  of the initial diverse sample set `m`.
- `nchange`: Number of consecutive iterations without improvement after which a 
  population member is considered trapped and is replaced by a random solution 
  (value of 22 determined experimentally in Section 3.1).
- `penalty`: Parameter `w` from Eq. 17, kept constant throughout the optimization. 
  Only used if the problem has constraints (`lcons`/`ucons`). Must be large 
  enough to dominate the scale of the objective function.
- `rng`: Random number generator instance.
"""
struct EACOP{R} <: SciMLBase.AbstractOptimizationAlgorithm
    pop_multiplier::Int
    nchange::Int
    penalty::Float64
    rng::R
end

EACOP(; pop_multiplier = 10, nchange = 22, penalty = 1e6,
        rng = Random.default_rng()) =
    EACOP(pop_multiplier, nchange, penalty, rng)

# ---------------------------------------------------------------------------
# 2. Utility Functions
# ---------------------------------------------------------------------------

"""
    population_size(nvar, mult)

Compute population size `b`: the first even number satisfying `b² - b ≥ mult * nvar`
(Table 1 of the article). A minimum size of 6 is enforced so that the bias β is 
well-defined (denominator `b - 2`).
"""
function population_size(nvar::Integer, mult::Integer)
    target = mult * nvar
    b = 6
    while b^2 - b < target
        b += 2
    end
    return b
end

"""
    combine!(xnew, pop, i, j, b, lb, ub, rng)

Generate an offspring solution within the hyper-rectangle constructed *around* 
member `i` and biased by the relative quality of `i` compared to `j` 
(Equations 9–14). The population matrix `pop` must be sorted by quality.
"""
function combine!(xnew, pop, i::Int, j::Int, b::Int, lb, ub, rng)
    α = i < j ? 1.0 : -1.0            # Eq. 12
    β = (abs(j - i) - 1) / (b - 2)    # Eq. 13
    @inbounds for k in eachindex(xnew)
        d  = (pop[k, j] - pop[k, i]) / 2          # Eq. 11
        c1 = pop[k, i] - d * (1 + α * β)          # Eq. 9
        c2 = pop[k, i] + d * (1 - α * β)          # Eq. 10
        r  = rand(rng)                            # Eq. 14
        xnew[k] = clamp(c1 + (c2 - c1) * r, lb[k], ub[k])
    end
    return xnew
end

"""
    go_beyond(f, xpr, fpr, xch, fch, lb, ub, rng)

Algorithm 1: As long as the offspring continues to outperform its parent, generate 
a new solution "beyond" in the parent → offspring direction. After two consecutive 
improvements, the generation area is doubled (`Λ ← Λ/2`).

Returns `(x, f)` of the best solution found along the trajectory.
"""
function go_beyond(f, xpr, fpr, xch, fch, lb, ub, rng)
    improvement = 1
    Λ = 1.0
    xnew = similar(xch)
    while fch < fpr
        # Rectangle defined by [xch - (xpr - xch)/Λ , xch]
        @inbounds for k in eachindex(xnew)
            lo = xch[k] - (xpr[k] - xch[k]) / Λ
            hi = xch[k]
            lo, hi = minmax(lo, hi)
            xnew[k] = clamp(lo + (hi - lo) * rand(rng), lb[k], ub[k])
        end
        fnew = f(xnew)

        xpr, fpr = xch, fch  # Offspring becomes the new parent
        xch, fch = copy(xnew), fnew  # New solution becomes the new offspring

        improvement += 1
        if improvement == 2
            Λ /= 2           # Double the search area
            improvement = 0
        end
    end
    # Upon exiting the loop, `xpr` holds the last solution that successfully improved.
    return xpr, fpr
end

# ---------------------------------------------------------------------------
# 3. SciMLBase.solve Overload
# ---------------------------------------------------------------------------

function SciMLBase.solve(prob::SciMLBase.OptimizationProblem, alg::EACOP;
                         maxiters = 1000, maxfevals = typemax(Int), kwargs...)

    lb, ub = prob.lb, prob.ub
    (lb === nothing || ub === nothing) &&
        throw(ArgumentError("EACOP requires finite bounds `lb` and `ub`."))

    t0 = time()

    nvar = length(lb)
    rng  = alg.rng
    lb   = collect(float.(lb))
    ub   = collect(float.(ub))

    # --- Optimization Sense --------------------------------------------------
    # EACOP always minimizes; with MaxSense, -C(x) is minimized internally.
    sgn = (prob.sense === SciMLBase.MaxSense) ? -1.0 : 1.0

    # --- Constraints: Static Penalty (Eq. 17) --------------------------------
    has_cons = hasproperty(prob.f, :cons) && (prob.f.cons !== nothing) &&
               (prob.lcons !== nothing) && (prob.ucons !== nothing)
    ncons    = has_cons ? length(prob.lcons) : 0
    cons_buf = zeros(ncons)
    lcons    = has_cons ? collect(float.(prob.lcons)) : Float64[]
    ucons    = has_cons ? collect(float.(prob.ucons)) : Float64[]

    # L-∞ norm of constraint violations (0.0 if the point is feasible)
    function violation(x)
        has_cons || return 0.0
        prob.f.cons(cons_buf, x, prob.p)
        v = 0.0
        @inbounds for i in 1:ncons
            # Equality constraints are expressed as lcons[i] == ucons[i]
            vi = max(lcons[i] - cons_buf[i], cons_buf[i] - ucons[i], 0.0)
            v  = max(v, vi)
        end
        return v
    end

    # Original, un-transformed objective (used for reporting the final result)
    obj(x) = prob.f(x, prob.p)

    # Internal objective: F(x) = ±C(x) + w * max{viol} with function evaluation counter
    fevals = Ref(0)
    function f(x)
        fevals[] += 1
        return sgn * obj(x) + alg.penalty * violation(x)
    end

    # --- A. Initial Population -----------------------------------------------
    b = population_size(nvar, alg.pop_multiplier)
    m = max(alg.pop_multiplier * nvar, b)

    S  = QuasiMonteCarlo.sample(m, lb, ub, LatinHypercubeSample())   # nvar × m
    fS = [f(view(S, :, k)) for k in 1:m]

    order = sortperm(fS)
    half  = b ÷ 2
    best_idx = order[1:half]                                       # Half selected by quality
    rest     = order[half+1:end]
    rand_idx = rest[randperm(rng, length(rest))[1:half]]       # Half selected randomly
    sel      = vcat(best_idx, rand_idx)

    pop  = Matrix{Float64}(S[:, sel])
    fpop = fS[sel]
    nstuck = zeros(Int, b)

    # Global best solution
    kbest  = argmin(fpop)
    best_x = copy(pop[:, kbest])
    best_f = fpop[kbest]

    # Reusable buffers
    xnew  = zeros(nvar)
    child = zeros(nvar)

    iter = 0
    retcode = SciMLBase.ReturnCode.MaxIters

    while iter < maxiters
        iter += 1

        # Sort population by quality (required for Eqs. 12–13)
        p = sortperm(fpop)
        pop, fpop, nstuck = pop[:, p], fpop[p], nstuck[p]

        newpop  = copy(pop)
        newfpop = copy(fpop)
        labeled = falses(b)

        for i in 1:b
            # --- B. Combination: b - 1 offspring around member i -------------
            fchild = Inf
            for j in 1:b
                j == i && continue
                combine!(xnew, pop, i, j, b, lb, ub, rng)
                fx = f(xnew)
                if fx < fchild
                    fchild = fx
                    copyto!(child, xnew)
                end
            end

            # --- C. (1+1) Update + Go-Beyond ---------------------------------
            if fchild < fpop[i]
                xg, fg = go_beyond(f, view(pop, :, i), fpop[i],
                                   copy(child), fchild, lb, ub, rng)
                newpop[:, i] = xg
                newfpop[i]   = fg
                labeled[i]   = true
            end

            fevals[] ≥ maxfevals && break
        end

        pop, fpop = newpop, newfpop

        # --- D. Local Optima Escape Mechanism --------------------------------
        for i in 1:b
            if labeled[i]
                nstuck[i] = 0
            else
                nstuck[i] += 1
                if nstuck[i] > alg.nchange
                    @inbounds for k in 1:nvar
                        pop[k, i] = lb[k] + (ub[k] - lb[k]) * rand(rng)
                    end
                    fpop[i]   = f(view(pop, :, i))
                    nstuck[i] = 0
                end
            end
        end

        # Update global best
        kbest = argmin(fpop)
        if fpop[kbest] < best_f
            best_f = fpop[kbest]
            copyto!(best_x, view(pop, :, kbest))
        end

        if fevals[] ≥ maxfevals
            retcode = SciMLBase.ReturnCode.MaxIters
            break
        end
    end

    # --- E. Package Result ---------------------------------------------------
    # Report the *original* objective value (without sign inversion or penalty);
    # `best_f` represents the internal minimized value.
    best_obj  = obj(best_x)
    best_viol = violation(best_x)

    if has_cons && best_viol > 0
        retcode = INFEASIBLE
    end

    stats = SciMLBase.OptimizationStats(; iterations = iter,
                                          fevals = fevals[],
                                          time = time() - t0)

    return SciMLBase.build_solution(EACOPCache(prob), alg, best_x, best_obj;
                                    retcode = retcode,
                                    stats = stats,
                                    original = (; popsize = b,
                                                penalized_objective = best_f,
                                                max_violation = best_viol))
end

end # module
