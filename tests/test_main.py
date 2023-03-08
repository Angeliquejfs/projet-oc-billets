# Creer focntion test
from main import multiply_by_ten


def test_multiply_by_ten():
    """
    Cette fonction test la fonction de multiplication par 10.
    """
    assert multiply_by_ten(3) == 15
    assert multiply_by_ten(10) == 100
