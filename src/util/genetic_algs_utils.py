import random

ONEMAX_GENOTYPE_LENGTH = 10

def onemax_random_genotype() -> int:
	return random.randint(0, 2 ** ONEMAX_GENOTYPE_LENGTH - 1)

def onemax_create_population(population_size: int) -> list[int]:
    return [onemax_random_genotype() for _ in range(population_size)]
