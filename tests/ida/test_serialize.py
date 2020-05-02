import unittest
from typing import List
import random

from src.ida.as_json import AsJson


class A(AsJson):
    def __init__(self):
        self.name: str = 'string'
        self.age: int = 10

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age


class B(AsJson):
    a: A

    def __init__(self):
        self.a = A()

    def __eq__(self, other):
        return self.a == other.a


class C(AsJson):
    a: A
    c: List[A]

    def __init__(self):
        self.a = A()
        self.c = []

    def __eq__(self, other):
        t = self.a == other.a
        for l, r in zip(self.c, other.c):
            t = t and l == r
        return t


def get_a():
    names = ['hello', 'limo', 'cook']
    ages = [10, 2, 44]
    a = A()
    a.name = random.choice(names)
    a.age = random.choice(ages)
    return a


def get_b():
    b = B()
    b.a = get_a()
    return b


def get_c():
    c = C()
    c.a = get_a()
    c.c = [get_a(), get_a(), get_a()]
    return c


def compare_a_and_d(a, dic):
    return a.name == dic['name'] and a.age == dic['age']


class TestSerialize(unittest.TestCase):
    def test_automatic_to_dict(self):
        b = get_b()
        js = AsJson.to_dict(b)
        self.assertTrue(compare_a_and_d(b.a, js['a']))

    def test_automatic_from_dict(self):
        b = get_b()
        dic = AsJson.to_dict(b)
        b = AsJson.from_dict(B, dic)
        self.assertTrue(compare_a_and_d(b.a, dic['a']))


class TestSerializeIter(unittest.TestCase):
    def test_to_from_dict_in_list(self):
        bs = [get_b() for _ in range(4)]
        jss = AsJson.to_dict(bs)

        for b, js in zip(bs, jss):
            self.assertTrue(compare_a_and_d(b.a, js['a']))

        bs_ = AsJson.from_dict(List[B], jss)
        self.assertEqual(bs, bs_)

    def test_to_from_dict_in_nest_list(self):
        bs = [[get_b() for _ in range(4)] for _ in range(3)]
        jss = AsJson.to_dict(bs)

        for b, js in zip(bs, jss):
            for i, j in zip(b, js):
                self.assertTrue(compare_a_and_d(i.a, j['a']))

        bs_ = AsJson.from_dict(List[List[B]], jss)
        self.assertEqual(bs, bs_)

    def test_to_dict_in_field_list(self):
        c = get_c()
        js = AsJson.to_dict(c)

        self.assertTrue(compare_a_and_d(c.a, js['a']))

        for i, j in zip(c.c, js['c']):
            self.assertTrue(compare_a_and_d(i, j))
