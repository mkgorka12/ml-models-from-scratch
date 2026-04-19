import random

def onemax_select_best(genotypes: list[int], fitnesses: list[float]) -> list[int]:
	genotypes_fitnesses_sorted = sorted(zip(genotypes, fitnesses), key=lambda x: x[1], reverse=True)

	genotypes_sorted = [x[0] for x in genotypes_fitnesses_sorted]
	return genotypes_sorted
