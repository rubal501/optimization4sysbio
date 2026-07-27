"""
    EACOP.jl

Implementación sencilla del *Evolutionary Algorithm for Complex-process Optimization*
(Egea, Martí & Banga, 2009), compatible con la interfaz de SciMLBase.

Componentes del artículo implementados:
  * Población inicial por muestreo de hipercubo latino (mitad por calidad, mitad aleatoria).
  * Combinación mediante hiper-rectángulos anchos y sesgados (ecs. 9-14).
  * Actualización (1+1): un hijo solo entra reemplazando a su propio padre.
  * Estrategia *go-beyond* (Algoritmo 1).
  * Escape de óptimos locales mediante el contador `nstuck` / parámetro `nchange`.
  * Restricciones tratadas con el término de penalización estática de la ec. 17
    (norma L-∞ de las violaciones).

Además se respeta `prob.sense`: con `MaxSense` se minimiza internamente `-C(x)`.

No incluye búsqueda local, tal y como se discute en la Sección 1 del artículo.
"""
module EACOPOptimizer

import SciMLBase
using QuasiMonteCarlo
using Random

export EACOP

# `Infeasible` no existe en versiones antiguas de SciMLBase
const INFEASIBLE = isdefined(SciMLBase.ReturnCode, :Infeasible) ?
                   SciMLBase.ReturnCode.Infeasible : SciMLBase.ReturnCode.Failure

"""
    EACOPCache(prob)

Caché mínima que satisface la interfaz de `SciMLBase.AbstractOptimizationCache`.
`build_solution` solo admite una caché como primer argumento (no un
`OptimizationProblem`), y accede a `cache.p` y `cache.f` para el indexado
simbólico y los observables. Se guarda también el problema por comodidad.
"""
struct EACOPCache{F, P, PR} <: SciMLBase.AbstractOptimizationCache
    f::F
    p::P
    prob::PR
end

EACOPCache(prob) = EACOPCache(prob.f, prob.p, prob)

# ---------------------------------------------------------------------------
# 1. Definición del algoritmo y sus hiperparámetros
# ---------------------------------------------------------------------------

"""
    EACOP(; pop_multiplier = 10, nchange = 22, penalty = 1e6, rng = Random.default_rng())

- `pop_multiplier`: número aproximado de soluciones nuevas generadas por iteración,
  por variable de decisión (10·nvar en el artículo). Determina también el tamaño
  del conjunto diverso inicial `m`.
- `nchange`: número de iteraciones consecutivas sin mejora tras las cuales un
  miembro de la población se considera atrapado y se reemplaza por una solución
  aleatoria (valor 22 determinado experimentalmente en la Sección 3.1).
- `penalty`: parámetro `w` de la ec. 17, constante durante toda la optimización.
  Solo se usa si el problema tiene restricciones (`lcons`/`ucons`). Debe ser lo
  bastante grande como para dominar la escala del objetivo.
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
# 2. Utilidades
# ---------------------------------------------------------------------------

"""
    population_size(nvar, mult)

Tamaño de población `b`: el primer número par que cumple `b² - b ≥ mult·nvar`
(Tabla 1 del artículo). Se impone un mínimo de 6 para que el sesgo β esté bien
definido (denominador `b - 2`).
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

Genera una solución dentro del hiper-rectángulo construido *alrededor* del
miembro `i` y sesgado por la calidad relativa de `i` frente a `j`
(ecuaciones 9-14). La población debe estar ordenada por calidad.
"""
function combine!(xnew, pop, i::Int, j::Int, b::Int, lb, ub, rng)
    α = i < j ? 1.0 : -1.0            # ec. 12
    β = (abs(j - i) - 1) / (b - 2)    # ec. 13
    @inbounds for k in eachindex(xnew)
        d  = (pop[k, j] - pop[k, i]) / 2          # ec. 11
        c1 = pop[k, i] - d * (1 + α * β)          # ec. 9
        c2 = pop[k, i] + d * (1 - α * β)          # ec. 10
        r  = rand(rng)                            # ec. 14
        xnew[k] = clamp(c1 + (c2 - c1) * r, lb[k], ub[k])
    end
    return xnew
end

"""
    go_beyond(f, xpr, fpr, xch, fch, lb, ub, rng)

Algoritmo 1: mientras el hijo siga superando a su padre, se crea una nueva
solución "más allá" en la dirección padre → hijo. Tras dos mejoras consecutivas
se duplica el área de generación (Λ ← Λ/2).

Devuelve `(x, f)` de la mejor solución encontrada en la cadena.
"""
function go_beyond(f, xpr, fpr, xch, fch, lb, ub, rng)
    improvement = 1
    Λ = 1.0
    xnew = similar(xch)
    while fch < fpr
        # Rectángulo definido por [xch - (xpr - xch)/Λ , xch]
        @inbounds for k in eachindex(xnew)
            lo = xch[k] - (xpr[k] - xch[k]) / Λ
            hi = xch[k]
            lo, hi = minmax(lo, hi)
            xnew[k] = clamp(lo + (hi - lo) * rand(rng), lb[k], ub[k])
        end
        fnew = f(xnew)

        xpr, fpr = xch, fch          # el hijo pasa a ser el padre
        xch, fch = copy(xnew), fnew  # la nueva solución es el nuevo hijo

        improvement += 1
        if improvement == 2
            Λ /= 2                   # duplica el área de búsqueda
            improvement = 0
        end
    end
    # Al salir del bucle, `xpr` es la última solución que sí mejoró.
    return xpr, fpr
end

# ---------------------------------------------------------------------------
# 3. Sobrecarga de solve
# ---------------------------------------------------------------------------

function SciMLBase.solve(prob::SciMLBase.OptimizationProblem, alg::EACOP;
                         maxiters = 1000, maxfevals = typemax(Int), kwargs...)

    lb, ub = prob.lb, prob.ub
    (lb === nothing || ub === nothing) &&
        throw(ArgumentError("EACOP requiere cotas finitas `lb` y `ub`."))

    t0 = time()

    nvar = length(lb)
    rng  = alg.rng
    lb   = collect(float.(lb))
    ub   = collect(float.(ub))

    # --- Sentido de la optimización ------------------------------------------
    # EACOP minimiza siempre; con MaxSense se minimiza internamente -C(x).
    sgn = (prob.sense === SciMLBase.MaxSense) ? -1.0 : 1.0

    # --- Restricciones: penalización estática (ec. 17) ------------------------
    has_cons = hasproperty(prob.f, :cons) && (prob.f.cons !== nothing) &&
               (prob.lcons !== nothing) && (prob.ucons !== nothing)
    ncons    = has_cons ? length(prob.lcons) : 0
    cons_buf = zeros(ncons)
    lcons    = has_cons ? collect(float.(prob.lcons)) : Float64[]
    ucons    = has_cons ? collect(float.(prob.ucons)) : Float64[]

    # Norma L-∞ de la violación de restricciones (0 si el punto es factible)
    function violation(x)
        has_cons || return 0.0
        prob.f.cons(cons_buf, x, prob.p)
        v = 0.0
        @inbounds for i in 1:ncons
            # Las igualdades se expresan como lcons[i] == ucons[i]
            vi = max(lcons[i] - cons_buf[i], cons_buf[i] - ucons[i], 0.0)
            v  = max(v, vi)
        end
        return v
    end

    # Objetivo original, sin transformar (para reportar el resultado)
    obj(x) = prob.f(x, prob.p)

    # Objetivo interno: F(x) = ±C(x) + w · max{viol}   con contador de evaluaciones
    fevals = Ref(0)
    function f(x)
        fevals[] += 1
        return sgn * obj(x) + alg.penalty * violation(x)
    end

    # --- A. Población inicial -------------------------------------------------
    b = population_size(nvar, alg.pop_multiplier)
    m = max(alg.pop_multiplier * nvar, b)

    S  = QuasiMonteCarlo.sample(m, lb, ub, LatinHypercubeSample())   # nvar × m
    fS = [f(view(S, :, k)) for k in 1:m]

    order = sortperm(fS)
    half  = b ÷ 2
    best_idx = order[1:half]                                  # mitad por calidad
    rest     = order[half+1:end]
    rand_idx = rest[randperm(rng, length(rest))[1:half]]       # mitad aleatoria
    sel      = vcat(best_idx, rand_idx)

    pop  = Matrix{Float64}(S[:, sel])
    fpop = fS[sel]
    nstuck = zeros(Int, b)

    # Mejor solución global
    kbest  = argmin(fpop)
    best_x = copy(pop[:, kbest])
    best_f = fpop[kbest]

    # Buffers reutilizables
    xnew  = zeros(nvar)
    child = zeros(nvar)

    iter = 0
    retcode = SciMLBase.ReturnCode.MaxIters

    while iter < maxiters
        iter += 1

        # Ordenar la población por calidad (requisito de las ecs. 12-13)
        p = sortperm(fpop)
        pop, fpop, nstuck = pop[:, p], fpop[p], nstuck[p]

        newpop  = copy(pop)
        newfpop = copy(fpop)
        labeled = falses(b)

        for i in 1:b
            # --- B. Combinación: b-1 hijos alrededor del miembro i -----------
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

            # --- C. Actualización (1+1) + go-beyond --------------------------
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

        # --- D. Escape de óptimos locales -----------------------------------
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

        # Actualizar el mejor global
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

    # --- E. Empaquetar el resultado ------------------------------------------
    # Se reporta el valor *original* del objetivo (sin signo invertido ni
    # penalización); `best_f` es el valor interno minimizado.
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
