from itertools import combinations_with_replacement

from sdfmpneo.training import late_stage_batch as batch


def test_direct_candidate_generator_preserves_general_response_parent_budgets():
    base = ("a", "b", "c")
    response = ("r0", "r1", "r2")
    names = base + response
    response_set = set(response)
    for max_response in (0, 1, 2, 3):
        for degree in range(5):
            expected = [
                parents for parents in combinations_with_replacement(names, degree)
                if sum(parent in response_set for parent in parents) <= max_response
            ]
            actual = list(
                batch._iter_admissible_parent_tuples(
                    names, response_set, degree, max_response
                )
            )
            assert actual == expected
