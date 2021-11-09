import random
import numpy as np
import numdifftools as nd
import math


class Variable():
    def identity(values, name):
        # credit to Michael Huang for the idea to find gradients using the identity array!
        val = np.zeros((len(values)))
        idx = [i for i,key in enumerate(values.keys()) if key==name]
        if len(idx) == 0:
            raise ValueError(f'Cannot find key {name} in the input values')
        val[idx[0]] = 1
        return val

    def __init__(self, name=None, evaluate=None, grad=None, representation=None):
        self.identifier = hash(random.random())
        if evaluate == None:
            self.evaluate = lambda values: values[self.name]
        else:
            self.evaluate = evaluate
        if grad == None:
            self.grad = lambda values: Variable.identity(values, name)
        elif grad != None:
            self.grad = grad
        self.name = name
        self.representation = representation
    
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Variable(inputs = [self], evaluate = lambda *values: other + self.evaluate(*values), grad = self.grad, representation= "%s + %s"% ((str(self)), str(other)))
        elif isinstance(other, Variable):
            return Variable(inputs = [self, other], evaluate = lambda *values: self.evaluate(*values) + other.evaluate(*values), grad = lambda *values: self.grad(*values) + other.grad(*values), representation= "%s + %s"% ((str(self)), str(other)))
        else:
            return NotImplemented
    
    def sin(var):
        if isinstance(var, (int, float)):
            return np.sin(var)
        return Variable(evaluate= lambda values : np.sin(var.evaluate(values)), 
        grad= lambda values : np.cos(var.evaluate(values)) * var.grad(values), representation= lambda : f'sin({var})') 

    def cos(var):
        if isinstance(var, (int, float)):
            return np.cos(var)
        return Variable(evaluate= lambda values : np.cos(var.evaluate(values)), 
        grad= lambda values : -np.sin(var.evaluate(values)) * var.grad(values), representation= lambda : f'cos({var})')

    @staticmethod
    def log(var):
        if isinstance(var, (int, float)):
            return np.log(var)
        elif (isinstance(var, (Variable))):
            return Variable(evaluate= lambda values: np.log(var.evaluate(values)), grad= lambda values: 1/var.evaluate(values) * var.grad(values), representation= f'ln({var})')
        
    
    def grad(self, values):
        self_location = self.order(values)
        pre_self = self_location
        post_self = len(values) - 1 - self_location
        gradient_array = [0] * pre_self + [1] + [0] *post_self
        return np.array(gradient_array)
    
    def __sub__(self, other):
        return self + other * -1

    def __call__(self, **kwargs):
        return self.evaluate(kwargs)

    def __repr__(self):
        if self.representation != None:
            return self.representation()
        if self.name != None:
            return self.name
        return "<%s>" % str(hash(self))[:3]


    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Variable(evaluate=lambda values: self.evaluate(values) + other, grad=lambda values: self.grad(values), representation=lambda: "(%s + %s)" % (str(self), str(other)))
        return Variable(evaluate=lambda values: self.evaluate(values) + other.evaluate(values), grad=lambda values: self.grad(values) + other.grad(values), representation=lambda: "(%s + %s)" % (str(self), str(other)))                     
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return self * -1 + other

    def __mul__(self, other):
        if isinstance(other, Variable):
            return Variable(evaluate=lambda values: self.evaluate(values) * other.evaluate(values), grad=lambda values: self.grad(values)*other.evaluate(values) + other.grad(values)*self.evaluate(values), representation=lambda: "(%s * %s)" % (str(self), str(other)))
        if isinstance(other, (float, int)):
            return Variable(evaluate=lambda values: self.evaluate(values) * other, grad=lambda values: self.grad(values) * other, representation=lambda: "(%s * %s)" % (str(self), str(other)))

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
     return Variable(evaluate=lambda values: self.evaluate(values) ** other, 
     grad=lambda values: (other)*(self.evaluate(values) ** (other - 1))*self.grad(values), 
     representation=lambda: "(%s ** %s)" % (str(self), str(other)))
     
    @staticmethod
    def exp(var):
        if isinstance(var, Variable):
            return Variable(evaluate=lambda values: math.e ** var.evaluate(values),
            grad=lambda values: (math.e ** var.evaluate(values))*var.grad(values),
            representation=lambda: "(e ** %s)" % str(var))
        elif isinstance(var, (float, int)):
            return math.e ** var
    
    def __truediv__(self, other):
        reciprocal = other ** -1
        return self * reciprocal

    def __rtruediv__(self, other):
        reciprocal = self ** -1
        return self * reciprocal

    def order(self, values):
        order = sorted([hash(key) for key in values.keys()])
        return order.index(hash(self))