def onemax_fitness(genotype: int) -> float:
    return genotype.bit_count()
