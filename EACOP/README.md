Implementación de *Evolutionary Algorithm for Complex-process Optimization* (Egea, Martí & Banga, 2009), compatible con la interfaz de SciMLBase.

Componentes del artículo implementados:
  * Población inicial por muestreo de hipercubo latino (mitad por calidad, mitad aleatoria).
  * Combinación mediante hiper-rectángulos anchos y sesgados (ecs. 9-14).
  * Actualización (1+1): un hijo solo entra reemplazando a su propio padre.
  * Estrategia *go-beyond* (Algoritmo 1).
  * Escape de óptimos locales mediante el contador `nstuck` / parámetro `nchange`.
  * Restricciones tratadas con el término de penalización estática de la ec. 17
    (norma L-∞ de las violaciones).


# TODO:
