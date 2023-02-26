import copy
import inspect
from typing import Optional, TypeVar, Type, Union

from yacs.config import CfgNode as CN

from .cfgNode import merge_a_into_b_builder, purge_builder_cfg, is_builder_cfg, \
    _assert_not_necessary, _assert_with_logging, BUILDERS, \
    merge_inputs

CFG_KEYWORD = 'cfg'

T = TypeVar('T', bound='Configurable')


def as_builder(cls: T):
    cls.FACTORY = {}
    name = cls.get_name()
    import logging
    if name in BUILDERS:
        logging.warning(f"{cls}'s name {name} has been "
                        f"registered as class {BUILDERS[name]}")
    BUILDERS[cls.get_name()] = cls
    cls.FACTORY[name] = cls
    return cls


def reconfig(func):
    # used to reconfig the functions..
    signature = inspect.signature(func)
    args = {k: v.default for k, v in signature.parameters.items()}
    raise NotImplementedError
    return func


def match_inputs(signature, *args, **kwargs):
    # TODO: find better ways..
    # Given args, and kwargs to the methods, we need to figure out
    # the one that has an explicit input, e.g., cfg, the one in kwargs, or the one in args ..
    # print(signature.parameters)
    new_args = []
    args = list(args)
    num_args = 0
    for (k, v) in signature.parameters.items():
        if v.default != inspect.Parameter.empty:
            break
        num_args += 1
        if k in kwargs:
            # put default args in ..
            args.append(kwargs.pop(k))
    _assert_with_logging(len(args) >= num_args - 1,
                         lambda: f"Not enough args {args} for Signature: {str(signature.parameters)}")
    for idx, ((k, v), j) in enumerate(zip(signature.parameters.items(), [None, ] + args)):
        if idx == 0:
            continue
        if v.default == inspect.Parameter.empty:
            new_args.append(j)
        else:
            kwargs[k] = j
    if 'cfg' in kwargs:
        inp_cfg = kwargs.pop('cfg')
    else:
        inp_cfg = None
    return new_args, inp_cfg, kwargs


from yacs.config import _VALID_TYPES


def configurable_class(cls):
    # register cls into its first factory ancestor automatically ..
    p_factory = cls._get_factory()
    if p_factory is not None:
        p_factory[cls.get_name()] = cls

    signature = inspect.signature(cls.__init__)
    _cls_config = {k: v.default for k, v in signature.parameters.items()
                   if v.default is not inspect.Parameter.empty}

    _assert_not_necessary(CFG_KEYWORD in _cls_config, f"In {cls} with signature {signature}: please set cfg=None in the __init__ to make pycharm happier! "
                                                      "Do you forget to write __init__ for some functions?")
    _assert_not_necessary(_cls_config[CFG_KEYWORD] is None,
                          "Please set cfg=None in the __init__ to make pycharm happier!")
    del _cls_config[CFG_KEYWORD]

    for k, v in _cls_config.items():
        if isinstance(v, dict):
            _cls_config[k] = CN(v)

    for k, v in _cls_config.items():
        assert isinstance(v, CN) or type(v) in _VALID_TYPES, f"Default values can only be a CfgNode or {_VALID_TYPES}"

    for i, v in _cls_config.items():
        if isinstance(v, dict):
            _assert_not_necessary(isinstance(v, CN),
                                  f"Configurable class can't have python dict as default parameter for {i} of class {cls}!")

    def configurable_init(init):
        """
        the new init function
        it will use input_cfg, kwargs to override the existing cfg
        the priority is
            - existing self._cfg
            - input cfg (file, dict, cfgNode)
            - kwargs (cfgNode or any single parameters)
            - opts like "FOO.BAR="

        you can imagine that each class has a default configuration file,
        everytime you call __init__ of the class or its ancestor, you are using the kwargs in __init__ to override it.
        the order you override it depends the order you initialize the classes ..
        we call elements in kwargs the explicit initializations.
        TODO: raise an warning if paremters are explicitly initialized twice!
        """

        def wrapper(self, *args, **kwargs):
            # step 1. merge all inputted configures
            # match parameters with __init__ to find cfg if provided
            args, inp_cfg, kwargs = match_inputs(cls.__signature__, *args, **kwargs)

            # TODO: remove parameters that is not a part of default_cfg and raise an error...
            cfg = merge_inputs(inp_cfg, **kwargs)
            # print(cls.get_name(), inp_cfg, cfg)
            # the key trick here is that a parameter with a classname as default value,
            #   it means that that parameter is a factory, and we can't decide its default config right now..
            if self._cfg is None:
                # get_default_config try to decide all factories's parameter if possible ..
                default_cfg = self.get_default_config()
            else:
                default_cfg = self._cfg

            cfg = merge_a_into_b_builder(cfg, default_cfg)
            self._cfg = cfg

            # override kwargs
            kwargs = {}
            for i, _ in _cls_config.items():
                v = copy.deepcopy(cfg[i])
                if isinstance(v, CN):
                    if is_builder_cfg(v):
                        v = purge_builder_cfg(v, level=1)  # only purge for one-level ...
                kwargs[i] = v
            init(self, *args, cfg=cfg, **kwargs)
            _assert_not_necessary(self._initialized,
                                  f"{cls} doesn't initialize its all ancestors... this is very dangerous..")

        return wrapper

    cls._sub_config = CN(_cls_config)
    cls.__signature__ = signature
    cls.__old_init__ = cls.__init__
    cls.__configurable_wrapper__ = configurable_init
    cls.__init__ = configurable_init(cls.__init__)


class Configurable(object):
    _cfg: Optional[CN] = None
    _sub_config: Optional[CN] = None
    _initialized: bool = False
    __signature__ = None

    NAME: Optional[str] = None
    FACTORY: Optional[dict] = None

    def __init__(self, cfg=None):
        # I really want to add self.__dict__.update(self._cfg)
        # But this seems like a nightmare for code completion
        self._initialized = True

    @classmethod
    def parent_class(cls) -> Type[T]:
        # TODO: handle multiple inheritance; understand what happens with getmro.
        return inspect.getmro(cls)[1]

    @classmethod
    def get_name(cls) -> str:
        if cls.NAME is not None:
            return cls.NAME
        return cls.__name__

    @classmethod
    def get_default_config(cls, **kwargs) -> CN:
        """
        :param cfg: input class;
        :return: the default configratuion of
        """
        if cls is Configurable:
            return CN()
        p = cls.parent_class()
        cfg = p.get_default_config()
        cfg.set_new_allowed(True)
        try:
            merge_a_into_b_builder(cls._sub_config, cfg)
        except Exception as e:
            raise Exception(f"Error during merge {p}'s default into {cls} in get default_config: " + str(e))
        cfg.set_new_allowed(False)
        if len(kwargs) > 0:
            cfg = merge_inputs(cfg, **kwargs)
        return cfg

    def get_config(self) -> CN:
        return copy.deepcopy(self._cfg)

    ################################################################################
    # builder methods..
    ################################################################################

    @classmethod
    def is_builder(cls) -> bool:
        return cls.FACTORY is not None

    @classmethod
    def _get_factory(cls) -> dict:
        if cls is Configurable or cls.FACTORY is not None:
            return cls.FACTORY
        return cls.parent_class()._get_factory()

    @classmethod
    def get_type_instance(cls, TYPE: str) -> Type[T]:
        f = cls._get_factory()
        _assert_with_logging(TYPE in f, f"{TYPE} not in {cls.get_name()}'s factory")
        return f[TYPE]

    @classmethod
    def to_build(cls, inp_cfg: Optional[Union[str, dict, CN]] = None,
                 TYPE: Optional[str] = None, EXPAND: bool = False, **kwargs) -> CN:
        # TODO: make the return immutable ..
        _assert_with_logging(cls.is_builder(), "You can only build a Factory class.")
        if TYPE is not None:
            if inspect.isclass(TYPE):
                TYPE = TYPE.get_name()
            _assert_with_logging(TYPE in cls._get_factory(), f"{TYPE} is not in the factory of class {cls}")
        cfg = CN(new_allowed=True)
        cfg.defrost()
        if TYPE is not None:
            cfg.TYPE = TYPE
        if EXPAND:
            _assert_with_logging(TYPE is not None, "One can't set expand default configuration without set TYPE")
            merge_a_into_b_builder(cls.get_type_instance(TYPE).get_default_config(), cfg)
        cfg.CLASS = cls.get_name()
        cfg.freeze()
        cfg.set_new_allowed(False)

        #print(inp_cfg, cfg)

        inp_cfg = merge_inputs(inp_cfg, **kwargs)
        merge_a_into_b_builder(inp_cfg, cfg)
        return cfg

    def __init_subclass__(cls, **kwargs):
        configurable_class(cls)

    @classmethod
    def build(cls, *args, **kwargs) -> T:
        assert cls.is_builder()

        args, cfg, kwargs = match_inputs(cls.__signature__, *args, **kwargs)
        cfg = merge_inputs(cfg, **kwargs)

        type = None
        if hasattr(cfg, "TYPE"):
            type = cfg.pop('TYPE')
        if hasattr(cfg, "CLASS"):
            __cls = cfg.pop('CLASS')
            _assert_with_logging(__cls == cls.get_name(),
                                 f"Build a {cls} with a factory cfgNode with a class name {__cls}!!")
        if 'TYPE' in kwargs:
            type = kwargs.pop('TYPE')

        _assert_with_logging(type is not None, f"Please input TYPE for the builder {cls} with name {cls.get_name()}.")
        new_type = cls.FACTORY[type]
        # print('build', new_type, cfg,'\n', args)
        return new_type(*args, cfg=cfg)

    def __str__(self):
        return "#" * 80 + f"\nCLASS {self.get_name()}\n" + str(self.get_config()) + '\n' + "#" * 80

    @classmethod
    def __add__(self, other):
        return other

    @classmethod
    def purge(self, cfg, level=100, strict=False, inp_cfg=None, **kwargs):
        a = merge_a_into_b_builder(merge_inputs(inp_cfg, **kwargs), cfg)
        return purge_builder_cfg(a, level, strict=strict)


    @classmethod
    def parse(cls, *args,cfg=None, parser=None, parse_prefix=None, **kwargs):
        from .parse_args import parse_args
        if cfg is not None:
            assert isinstance(cfg, str), "for argument parser please input cfg as a file path"
        wrapper = parse_args(cfg, parser=parser, parse_prefix=parse_prefix)
        old_init = cls.__init__
        cls.__init__ = cls.__configurable_wrapper__(wrapper(cls.__old_init__))
        obj = cls(*args, **kwargs)
        cls.__init__ = old_init
        return obj