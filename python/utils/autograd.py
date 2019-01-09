import math


class op:
    def eval(self, values):
        pass

    def grad(self, values, over):
        pass

    def __call__(self, *args, **kwargs):
        return self.eval(args[0])


class symbol(op):
    def __init__(self, name):
        self.name = name

    def eval(self, values):
        return values[self]

    def grad(self, values, over):
        if over == self:
            return 1
        return 0


class constant(op):
    def __init__(self, a):
        # z = a
        # z' = 0
        self.a = a

    def eval(self, values):
        return self.a

    def grad(self, values, over):
        return 0


class minus(op):
    def __init__(self, a):
        # z = -a
        # z' = -a'
        self.a = a

    def eval(self, values):
        a = self.a(values)
        return -a

    def grad(self, values, over):
        da = self.a.grad(values, over)
        return -da


class exp(op):
    def __init__(self, a):
        # z = exp(a)
        # z'= exp(a) * a'
        self.a = a

    def eval(self, values):
        a = self.a(values)
        return math.exp(a)

    def grad(self, values, over):
        a = self.a(values)
        da = self.a.grad(values, over)
        return math.exp(a) * da


class add(op):
    def __init__(self, a, b):
        # z = a + b
        # z' = a' + b'
        self.a = a
        self.b = b

    def eval(self, values):
        a = self.a(values)
        b = self.b(values)
        return a + b

    def grad(self, values, over):
        da = self.a.grad(values, over)
        db = self.b.grad(values, over)
        return da + db


class sin(op):
    def __init__(self, a):
        # z = sin(a)
        # z' = cos(a) * a'
        self.a = a

    def eval(self, values):
        a = self.a(values)
        return math.sin(a)

    def grad(self, values, over):
        a = self.a(values)
        da = self.a.grad(values, over)
        return math.cos(a) * da


class mul(op):
    def __init__(self, a, b):
        # z = a * b
        # z' = a' * b + a * b'
        self.a = a
        self.b = b

    def eval(self, values):
        a = self.a(values)
        b = self.b(values)
        return a * b

    def grad(self, values, over):
        a = self.a(values)
        b = self.b(values)
        da = self.a.grad(values, over)
        db = self.b.grad(values, over)
        return da * b + a * db


class div(op):
    def __init__(self, a, b):
        # z = a / b
        # z' = (a' * b - a * b') / b^2
        self.a = a
        self.b = b

    def eval(self, values):
        a = self.a(values)
        b = self.b(values)
        return a / b

    def grad(self, values, over):
        a = self.a(values)
        b = self.b(values)
        da = self.a.grad(values, over)
        db = self.b.grad(values, over)
        return (da * b - a * db) / math.pow(b, 2)


x = symbol('x')
y = symbol('y')

values = {x: 2, y: 3}

sigmoid = div(constant(1), add(constant(1), exp(minus(x))))

for a in range(-600, 600):
    a /= 100.0
    print(sigmoid({x: a}), sigmoid.grad({x: a}, x))

