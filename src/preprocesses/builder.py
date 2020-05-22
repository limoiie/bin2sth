class ModelBuilder:
    __register = {}
    __register_raw = {}

    @staticmethod
    def clazz(Cls):
        return ModelBuilder.__register.get(Cls)

    @staticmethod
    def model_class(name):
        return ModelBuilder.__register_raw[name]

    @staticmethod
    def register(RawCls):
        def f(Builder):
            if RawCls in ModelBuilder.__register:
                raise RuntimeError(f'A Builder has already been registered for'
                                   f'{RawCls}!')
            ModelBuilder.__register[RawCls] = Builder
            ModelBuilder.register_cls(RawCls)
            return Builder
        return f

    @staticmethod
    def register_cls(Cls):
        if Cls.__name__ in ModelBuilder.__register_raw:
            raise RuntimeError(f'{Cls} already been registered!')
        ModelBuilder.__register_raw[Cls.__name__] = Cls
        return Cls

    def build(self, *args, **kwargs):
        raise NotImplementedError()


class ModelKeeper:
    __register = {}

    @staticmethod
    def instance(Cls):
        if Cls not in ModelKeeper.__register:
            raise ModuleNotFoundError(f'No Keeper is registered for {Cls}!')
        return ModelKeeper.__register[Cls]

    @staticmethod
    def register(RawCls):
        def f(SubKeeper):
            if RawCls in ModelKeeper.__register:
                raise RuntimeError(f'A Keeper has already been registered for'
                                   f'{RawCls}!')
            ModelKeeper.__register[RawCls] = SubKeeper()
            return SubKeeper
        return f

    def from_state(self, state):
        raise NotImplementedError()

    def state(self, model):
        raise NotImplementedError()


class WholeModelKeeper(ModelKeeper):
    def from_state(self, state):
        return state

    def state(self, model):
        return model
