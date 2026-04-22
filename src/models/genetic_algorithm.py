from typing import Callable, Any
import copy
import random


class GeneticAlgorithm:
    def __init__(
        self,
        *,
        create_population_func: Callable[[int], list[Any]],
        fitness_func: Callable[[Any], float],
        selection_func: Callable[[list[Any], list[float]], list],
        crossover_func: Callable[[Any, Any], tuple[Any, Any]],
        mutation_func: Callable[[Any], Any],
        mutation_probability: float = 0.05,
        elitism_count: int = 0,
    ):
        self.create_population = create_population_func
        self.fitness = fitness_func
        self.selection = selection_func
        self.crossover = crossover_func
        self.mutation = mutation_func
        self.mutation_probability = mutation_probability
        self.elitism_count = elitism_count

    def _get_elites(
        self, genotypes: list[Any], fitnesses: list[float]
    ) -> tuple[list[Any], list[Any], list[Any]]:
        genotypes_with_fitnesses = zip(genotypes, fitnesses)
        genotypes_with_fitnesses_sorted = sorted(
            genotypes_with_fitnesses, key=lambda x: x[1]
        )

        elites = []
        for _ in range(self.elitism_count):
            elites.append(genotypes_with_fitnesses_sorted[0][0])
            genotypes_with_fitnesses_sorted.pop(0)

        non_elite_genotypes, not_elite_fitnesses = zip(*genotypes_with_fitnesses_sorted)
        return elites, non_elite_genotypes, not_elite_fitnesses

    def _crossover(self, genotypes: list[Any]) -> list[Any]:
        new_genotypes = copy.deepcopy(genotypes)

        for _ in range(len(new_genotypes) // 2):
            parent1 = new_genotypes.pop()
            parent2 = new_genotypes.pop()

            children1, children2 = self.crossover(parent1, parent2)

            new_genotypes.insert(0, children1)
            new_genotypes.insert(0, children2)

        return new_genotypes

    def _mutation(self, genotypes: list[Any]) -> list[Any]:
        new_genotypes = copy.deepcopy(genotypes)

        for idx in range(len(new_genotypes)):
            mutation_propability = random.random()
            if mutation_propability <= self.mutation_probability:
                new_genotypes[idx] = self.mutation(new_genotypes[idx])

        return new_genotypes

    def _get_compliants(
        self, genotypes: list[Any], fitnesses: list[float], acceptable_fitness: float
    ):
        genotypes_with_fitnesses = zip(genotypes, fitnesses)
        return [
            genotype
            for genotype, fitness in genotypes_with_fitnesses
            if fitness >= acceptable_fitness
        ]

    def run(
        self, population_size: int, acceptable_fitness: float, verbose=True
    ) -> tuple[list[Any], list[Any]]:
        genotypes = self.create_population(population_size)
        if verbose:
            print(genotypes)

        i = 0
        while True:
            fitnesses = [self.fitness(genotype) for genotype in genotypes]

            compliants = self._get_compliants(genotypes, fitnesses, acceptable_fitness)
            if len(compliants):
                break

            elites, genotypes, fitnesses = self._get_elites(genotypes, fitnesses)

            genotypes = self.selection(genotypes, fitnesses)
            genotypes = self._crossover(genotypes)
            genotypes = self._mutation(genotypes)

            genotypes += elites

            if verbose and i % 100 == 0:
                print("Iteration:", i, "Best fitness:", fitnesses[0])
            i += 1

        if verbose:
            print("Iterations:", i)

        return genotypes, compliants
