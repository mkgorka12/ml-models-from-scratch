from .genetic_algs_utils import ONEMAX_GENOTYPE_LENGTH

def onemax_single_point_crossing(genotype1: int, genotype2: int) -> tuple[int, int]:
	half_genotype_length = ONEMAX_GENOTYPE_LENGTH // 2

	mask_tail = 2 ** half_genotype_length - 1
	mask_head = mask_tail << half_genotype_length

	child1 = (genotype1 & mask_head) | (genotype2 & mask_tail)
	child2 = (genotype2 & mask_head) | (genotype1 & mask_tail)

	return child1, child2
