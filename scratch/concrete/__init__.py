class MockFhe:
    class Circuit:
        pass
    class Compiler:
        pass
    class Configuration:
        pass
    def LookupTable(self, *args, **kwargs):
        pass
    def array(self, *args, **kwargs):
        pass
    def multivariate(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator

fhe = MockFhe()
