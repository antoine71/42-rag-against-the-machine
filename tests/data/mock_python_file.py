import math as m
import re

TEST = 1234

for i in range(321):
    print(i)
    print("test")


class TestClass:
    """
    My test class
    """

    def __init__(self) -> None:
        self.a = m.sqrt(1)

    def test_test(self) -> float:
        return self.a


def test_fct() -> str:
    """
    My fct
    """
    re.compile(r"\d+", 1234)
    return "test return value"
