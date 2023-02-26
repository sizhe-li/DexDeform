import copy
import logging
from typing import Optional, Union

from yacs.config import CfgNode as CN, _check_and_coerce_cfg_value_type

################################################################################
# Logging...
################################################################################

logger = logging.getLogger(__name__)


def _assert_with_logging(cond, msg, suffix=''):
    if not cond:
        if not isinstance(msg, str):
            msg = msg()
        logger.debug(msg + suffix)
    assert cond, msg


def _assert_not_necessary(flag, x):
    _assert_with_logging(flag, x, "  This assertion is not necessary. Let me know if you feel unhappy with that! "
                                  "I want to know how many people will be happier without this limitation..")


def _assert_warning(flag, x):
    if flag:
        logger.warning(x)


################################################################################
# Traditional CfgNode
################################################################################

def _check_cfg_no_dot(cfg):
    for k, v in cfg.items():
        _assert_with_logging('.' not in k, lambda: f"Input cfg has an invalid key {k}")
        if isinstance(v, CN):
            _check_cfg_no_dot(v)

def load_v(v):
    if isinstance(v, str) and v[-4:] == 'yaml':
        logging.warning(f"loading yaml {v}!!!! This is a hack now!")
        cfg = CN(new_allowed=True)
        cfg.merge_from_file(v)
        return cfg
    return v

def merge_inputs(inp_cfg: Optional[Union[CN, dict, str]], **kwargs):
    # we ensure that inp_cfg and doesn't have factory node ...
    cfg = CN(new_allowed=True)
    if inp_cfg is not None:
        if isinstance(inp_cfg, str):
            cfg.merge_from_file(inp_cfg)
        elif isinstance(inp_cfg, dict):
            cfg.merge_from_other_cfg(CN(inp_cfg))
        else:
            from yacs.config import CfgNode
            assert isinstance(inp_cfg, CfgNode), f"input cfg is {type(inp_cfg)}: {cfg}"
            cfg.merge_from_other_cfg(inp_cfg)

    if len(kwargs) > 0:
        cn = CN(new_allowed=True)
        lists = []
        for i, v in kwargs.items():
            if '.' in i:
                lists.append(i)
                lists.append(v)
            else:
                cn[i] = load_v(v)
        #cfg.merge_from_other_cfg(cn)
        merge_a_into_b_builder(cn, cfg) #TODO: not sure if this is ok...

        if len(lists) > 0:
            for full_key, v in zip(lists[0::2], lists[1::2]):
                key_list = full_key.split(".")
                p = cfg
                for subkey in key_list[:-1]:
                    if not hasattr(p, subkey):
                        p.defrost()
                        cn = CN(new_allowed=True)
                        setattr(p, subkey, cn)
                    p = getattr(p, subkey)
                setattr(p, key_list[-1], load_v(v))

    _check_cfg_no_dot(cfg)
    cfg.set_new_allowed(False)
    cfg.freeze()
    return cfg


################################################################################
# Handle Builder
################################################################################

BUILDERS = {}


def is_builder_cfg(a: CN):
    return isinstance(a, CN) and hasattr(a, "CLASS")


def purge_builder_cfg(cfg: CN, level=1, strict=False):
    """
    Purge configuration,
    In another word... set up default configs for factory node with at most (level-1) factory parents..
    :param cfg:
    :param level:
    :param strict: if strict = False, we jump over
    :return:
    """
    assert level >= 1
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.defrost()
    if is_builder_cfg(cfg):
        if strict:
            _assert_with_logging(cfg.TYPE is not None,
                                 f"Purge failed..{cfg.CLASS} has None Type")

        if hasattr(cfg, "TYPE") and getattr(cfg, "TYPE") is not None:
            cls = BUILDERS[cfg.CLASS]
            default_cfg: CN = cls.get_type_instance(cfg.TYPE).get_default_config()
            default_cfg.defrost()
            default_cfg.TYPE = cfg.TYPE
            default_cfg.CLASS = cfg.CLASS
            default_cfg.freeze()
            cfg = merge_a_into_b_builder(cfg, default_cfg)
        level -= 1

    if level > 0:
        for i, v in cfg.items():
            if isinstance(v, CN):
                cfg[i] = purge_builder_cfg(v, level, strict)
    cfg.freeze()
    return cfg


def _check_builder_node_validity(a: CN):
    # out stores all parameters for which we can determine their default value.
    out, a_default = CN(new_allowed=True), CN(new_allowed=True)
    #print("#"*100)
    #print(a)
    #print("#"*100)
    if is_builder_cfg(a):
        # if the TYPE of this node is not decided, we don't check it as nothing could be determined yet..
        if not hasattr(a, "TYPE"):
            return None, None
        a_default.merge_from_other_cfg(BUILDERS[a.CLASS].get_type_instance(a.TYPE).get_default_config())
        #print(a.CLASS)
        #print(BUILDERS[a.CLASS].get_type_instance(a.TYPE))
        #print(BUILDERS[a.CLASS].get_type_instance(a.TYPE).get_default_config())
        #print('-'*100)
    else:
        #TODO: not sure if this is correct
        a_default.merge_from_other_cfg(a)


    for k, v in a.items():
        # do recursive check ..
        v_d = None
        #print("CHILD:", k, v)
        #if k in a_default:
        #    print('default', a_default[k])
        #print('k', k, 'v', v)
        if isinstance(v, CN):
            if k in a_default and a_default[k] is not None:
                c = copy.deepcopy(a_default[k])
                c.set_new_allowed(True)
                c.merge_from_other_cfg(v)
                v = c
            v, v_d = _check_builder_node_validity(v)
            #print('recurse', k, v)

        if v is not None and k != "TYPE" and k != "CLASS":
            out[k] = v
            # print(is_builder_cfg(a))
            # print(a, 'vvvvv', k, v, type(v),'\n', '--'*50, '\n', k, 'default', a_default,'\n', '--'*50, '\n', out, '\n')
            if v_d is not None:
                try:
                    p = a_default[k]
                except Exception as e:
                    #print(a)
                    #print('-'*60, k, '-'*60)
                    #print(v_d)
                    #print('-' * 122)
                    #print(a_default)
                    raise e
                p.set_new_allowed(True)
                p.merge_from_other_cfg(v_d)
                a_default[k] = p

    if is_builder_cfg(a):
        # do check here..
        #print('a', a, 'a_default', a_default)
        #print('out', out)
        a_default.freeze()
        out.freeze()
        d = copy.deepcopy(a_default)
        try:
            d.merge_from_other_cfg(out)
        except Exception as e:
            raise KeyError(f"Merging Factory Node Failed! " + str(e))

    a_default.set_new_allowed(False)
    out.set_new_allowed(False)
    return out, a_default


def merge_a_into_b_builder(a: CN, b: CN, key_list=[]):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.

    handles the factory node..
    """
    _assert_with_logging(
        isinstance(a, CN),
        "`a` (cur type {}) must be an instance of {}".format(type(a), CN),
    )
    _assert_with_logging(
        isinstance(b, CN),
        "`b` (cur type {}) must be an instance of {}".format(type(b), CN),
    )

    if not is_builder_cfg(b):
        # b is not a factory cfg..
        _assert_with_logging(not is_builder_cfg(a),
                             "Please don't merge a factory configuration into a non-factory configuration."
                             " Make sure that the type of parameters aligns with their ancestors!" + ('-'*60) + '\n'+ str(a) + '\n' + ('-'*60) + '\n' + str(b))
    else:
        pass
    if is_builder_cfg(a) and is_builder_cfg(b):
        _assert_with_logging(a.CLASS == b.CLASS, "Can't merge two factory node with different CLASS!")

    prev_new_allowed = b.is_new_allowed()
    if is_builder_cfg(b):
        b.set_new_allowed(True)

    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])

        v = copy.deepcopy(v_)
        v = b._decode_cfg_value(v)

        if k in b:
            if b[k] is not None:
                v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)
                # Recursively merge dicts
                if isinstance(v, CN):
                    try:
                        merge_a_into_b_builder(v, b[k], key_list + [k])
                    except BaseException:
                        raise
                else:
                    b[k] = v
            else:
                b[k] = v
        elif b.is_new_allowed():
            b[k] = v
        else:
            # we don't consider any deprecated or renamed key
            print(a, '\n', '-'*60, '\n', b)
            raise KeyError("Non-existent config key: {}".format(full_key))

    if is_builder_cfg(b):
        _check_builder_node_validity(b)
    b.set_new_allowed(prev_new_allowed)
    return b
