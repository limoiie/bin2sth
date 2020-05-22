from functools import wraps
from types import FunctionType


# reference from
# https://stackoverflow.com/questions/11349183/how-to-wrap-every-method-of-a-class

def wrapper(method):
    @wraps(method)
    def wrapped(*args, **kwrds):
        print('no pain no gain')
        #   ... <do something to/with "method" or the result of calling it>
        return method(*args, **kwrds)
    return wrapped


class MetaClass(type):
    def __new__(mcs, classname, bases, classDict):
        newClassDict = {}
        for attributeName, attribute in classDict.items():
            if isinstance(attribute, FunctionType):
                # replace it with a wrapped version
                attribute = wrapper(attribute)
            newClassDict[attributeName] = attribute
        newClassDict['trojan'] = lambda self: print('hello, i am trojan')
        return type.__new__(mcs, classname, bases, newClassDict)


class MyClass(metaclass=MetaClass):
    def method1(self):
        pass
