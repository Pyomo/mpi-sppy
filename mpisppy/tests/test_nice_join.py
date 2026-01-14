from unittest import TestCase

from mpisppy.utils import nice_join


class TestNiceJoin(TestCase):
    def test_nice_join_default(self):
        self.assertEqual(nice_join(["a", "b", "c"]), "a, b or c", "Should be 'a, b or c'")

    def test_nice_join_new_seperator(self):
        self.assertEqual(
            nice_join(["a", "b", "c"], separator="; "),
            "a; b or c",
            "Should use '; ' as separator.",
        )

    def test_nice_join_no_conjunction(self):
        self.assertEqual(
            nice_join(["a", "b", "c"], conjunction=None),
            "a, b, c",
            "Should use reproduce native join behavior.",
        )
        self.assertEqual(
            nice_join(["a"]),
            "a",
            "Shouldn't use a conjunction, because there are not enough elements.",
        )

    def test_nice_join_new_conjunction(self):
        self.assertEqual(
            nice_join(["a", "b", "c"], conjunction="and"),
            "a, b and c",
            "Should use 'and' as conjunction.",
        )

    def test_nice_join_with_single_string(self):
        self.assertEqual(
            nice_join("abc"),
            "a, b or c",
            "Should split string into substrings with length 1.",
        )

    def test_nice_join_with_warp_ins_single_quote_is_ture(self):
        self.assertEqual(
            nice_join(["a", "b", "c"], warp_in_single_quote=True),
            "'a', 'b' or 'c'",
            "Should wrap each item in single quotes.",
        )

    def test_nice_join_using_representation(self):
        class A:
            def __repr__(self) -> str:
                return self.__class__.__name__

        self.assertEqual(
            nice_join([A(), "b", "c"]),
            "A, b or c",
            "Should use the representation of class A().",
        )

    def test_nice_join_using_dictionary_as_as_iterator(self):
        d = {a: i for i, a in enumerate(list("abc"))}

        self.assertEqual(
            nice_join(d),
            "a, b or c",
            "Should join the keys of the dictionary.",
        )

        self.assertEqual(
            nice_join(d.keys()),
            "a, b or c",
            "Should join the keys of the dictionary.",
        )

        self.assertEqual(
            nice_join(d.values()),
            "0, 1 or 2",
            "Should join the values of the dictionary.",
        )

    def test_nice_raises_error(self):
        with self.assertRaises(TypeError) as e:
            nice_join(1)  # type: ignore
        self.assertEqual(
            str(e.exception),
            "'int' object is not iterable",
            "Should say 'object is not iterable'.",
        )
