from tools import Configurable, as_builder


@as_builder
class X(Configurable):
    def __init__(self, y_val=50, cfg=None):
        super(X, self).__init__()


class X1(X):
    def __init__(self, x_val=10, y_val=20, cfg=None):
        super(X1, self).__init__()


class X2(X):
    def __init__(self, x_val=20, cfg=None):
        super(X2, self).__init__()


@as_builder
class A(Configurable):
    def __init__(self, a_val='a', z_val=0, x=X.to_build(), cfg=None):
        super(A, self).__init__()
        self.x = X.build(cfg=x)


class A1(A):
    def __init__(self, a_val='a1', z_val=1, x=X.to_build(TYPE='X2', x_val=50), cfg=None):
        super(A1, self).__init__()


class A2(A):
    def __init__(self, a_val='a2', z_val=2, x=X.to_build(x_val=20), cfg=None):
        super(A2, self).__init__()


class A3(A2):
    def __init__(self, a_val='a3', x=X.to_build(TYPE='X1', y_val=10), cfg=None):
        super(A3, self).__init__()


def test_to_build():
    a = A(**{"x.TYPE": "X1"})
    print(a)
    print("Default\n", A.get_default_config())
    print("Purged\n", A.purge(A.get_default_config()))
    a1 = A1()
    print(a1)
    print("Default\n", A1.get_default_config())
    print("Purged\n", A1.purge(A1.get_default_config()))
    a2 = A2(**{"x.TYPE": "X1"})
    print(a2)
    a3 = A3()
    print(a3)


if __name__ == '__main__':
    test_to_build()
