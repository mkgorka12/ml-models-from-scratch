import random
from .genetic_algs_utils import ONEMAX_GENOTYPE_LENGTH


def onemax_mutation(genotype: int) -> int:
    mutation_at = random.randint(0, ONEMAX_GENOTYPE_LENGTH - 1)
    mask = 1 << mutation_at

    new_genotype = genotype ^ mask
    return new_genotype
