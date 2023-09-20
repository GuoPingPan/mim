# Copyright (c) OpenMMLab. All rights reserved.
import ast
import os
import os.path as osp
import shutil
import urllib.request
from datetime import datetime
from typing import Union

from mmengine import mkdir_or_exist
from mmengine.config import Config, ConfigDict
from mmengine.hub import get_config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from .common import MODULE2GitPACKAGE, __init__str, _import_pack_str
from .utils import *  # noqa: F401,F403
from .utils import (
    _replace_config_scope_to_pack,
    _transfer_to_export_import,
    _wrapper_all_registries_build_func,
    format_code,
    get_all_files,
)


def pack_tools(tool_name: str, scope: str, path: str, auto_import=False):
    """pack tools from web.

    Args:
        tool_name (str): the tool name in repo tools dir
        scope (str): the scope of repo
        path (str): the path to save tool
    """
    if os.path.exists(path):
        os.remove(path)

    try:
        web_pth = f'https://raw.githubusercontent.com/open-mmlab/' \
            f'{MODULE2GitPACKAGE[scope]}/main/tools/{tool_name}'

        urllib.request.urlretrieve(web_pth, path)
    except Exception as e:
        print(f'***********[ERROR:{e}] Network conditions is not good.'
              f' Reloading... *********** ')
        web_pth = f'https://gitee.com/guopingpan/{MODULE2GitPACKAGE[scope]}' \
            f'/raw/main/tools/{tool_name}'

        urllib.request.urlretrieve(web_pth, path)

    # automatically import the pack modules
    if auto_import:
        with open(path, 'r+') as f:
            lines = f.readlines()
            code = ''.join(lines[:1] + [_import_pack_str] + lines[1:])
            f.seek(0)
            f.write(code)
            f.truncate()


def export_from_cfg(cfg: Union[str, ConfigDict],
                    export_root_dir: str = None,
                    fast_test: bool = False):
    """A function to pack the minimum available package according to config
    file.

    Args:
        cfg  (:obj:`ConfigDict` or str): The config file for packing the
            minimum package.
        pack_root_dir (str): The pack directory to save the packed package.
    """
    # get config
    if isinstance(cfg, str):
        if '::' in cfg:
            cfg = get_config(cfg)
        else:
            cfg = Config.fromfile(cfg)

    default_scope = cfg.get('default_scope', 'mmengine')

    # automatically generate export_root_dir
    if export_root_dir is None:
        export_root_dir = f'pack_from_{default_scope}_' + datetime.now(
        ).strftime(r'%Y%m%d_%H%M%S')

    # export_module_dir
    export_module_dir = export_root_dir + '/pack'
    if osp.exists(export_module_dir):
        shutil.rmtree(export_module_dir)

    # export config
    if '.mim/' in cfg.filename:
        cfg_dir = osp.join(export_module_dir,
                           osp.dirname(cfg.filename.split('.mim/')[-1]))
        cfg_pth = osp.join(export_module_dir, cfg.filename.split('.mim/')[-1])
    else:
        cfg_dir = osp.join(export_module_dir, 'configs')
        cfg_pth = osp.join(cfg_dir, cfg.filename.split('/')[-1])

    mkdir_or_exist(cfg_dir)

    # NOTE: change parameters for fasterfast_ testing
    if fast_test:
        # for batch_norm using at least 2 data
        if 'dataset' in cfg.train_dataloader.dataset:
            cfg.train_dataloader.dataset.dataset.indices = [0, 1]
        else:
            cfg.train_dataloader.dataset.indices = [0, 1]
        cfg.train_dataloader.batch_size = 2

        if cfg.get('test_dataloader') is not None:
            cfg.test_dataloader.dataset.indices = [0, 1]
            cfg.test_dataloader.batch_size = 2

        if cfg.get('val_dataloader') is not None:
            cfg.val_dataloader.dataset.indices = [0, 1]
            cfg.val_dataloader.batch_size = 2

        if (cfg.train_cfg.get('type') == 'IterBasedTrainLoop') \
                or (cfg.train_cfg.get('by_epoch') is None):
            cfg.train_cfg.max_iters = 2
        else:
            cfg.train_cfg.max_epochs = 2

        cfg.train_cfg.val_interval = 1
        cfg.default_hooks.logger.interval = 1

        if 'param_scheduler' in cfg and cfg.param_scheduler is not None:
            if isinstance(cfg.param_scheduler, list):
                for lr_sc in cfg.param_scheduler:
                    lr_sc.begin = 0
                    lr_sc.end = 2
            else:
                cfg.param_scheduler.begin = 0
                cfg.param_scheduler.end = 2

    cfg.dump(cfg_pth)

    _replace_config_scope_to_pack(cfg_pth)

    # transform to default_scope
    init_default_scope(default_scope)

    # wrap ``Registry.build()`` for exporting modules
    _wrapper_all_registries_build_func(
        pack_module_dir=export_module_dir, scope=default_scope)

    cfg['work_dir'] = osp.join(export_root_dir,
                               'work_dirs')  # creat temp work_dirs for export

    # use runner to export all needed modules
    runner = Runner.from_cfg(cfg)
    runner.build_train_loop(cfg.train_cfg)
    if 'val_cfg' in cfg and cfg.val_cfg is not None:
        runner.build_val_loop(cfg.val_cfg)
    if 'test_cfg' in cfg and cfg.test_cfg is not None:
        runner.build_test_loop(cfg.test_cfg)
    if 'optim_wrapper' in cfg and cfg.optim_wrapper is not None:
        runner.optim_wrapper = runner.build_optim_wrapper(cfg.optim_wrapper)
    if 'param_scheduler' in cfg and cfg.param_scheduler is not None:
        runner.build_param_scheduler(cfg.param_scheduler)

    # add ``__init__.py`` to all dirs, for transferring directories
    # to be modules
    for directory, _, _ in os.walk(export_module_dir):
        if not osp.exists(osp.join(directory, '__init__.py')) \
                and 'configs' not in directory \
                and not directory.endswith(f'{export_module_dir}/'):
            with open(osp.join(directory, '__init__.py'), 'w') as f:
                f.write(__init__str)

    # get tools from web
    tools_dir = osp.join(export_root_dir, 'tools')
    os.makedirs(tools_dir, exist_ok=True)

    pack_tools(
        'train.py',
        scope=default_scope,
        path=osp.join(export_root_dir, 'tools/train.py'),
        auto_import=True)
    pack_tools(
        'test.py',
        scope=default_scope,
        path=osp.join(export_root_dir, 'tools/test.py'),
        auto_import=True)

    # postprocess for ``pack/registry.py``
    with open(
            osp.join(export_module_dir, 'registry.py'), encoding='utf-8') as f:
        ast_tree = ast.parse(f.read())

    for node in ast.walk(ast_tree):
        # check the location path for Registry to load modules
        # if the path hasn't been exported, then need to be removed.
        # finally will use the root Registry to find module until it
        # actually doesn't exist.
        if isinstance(node, ast.Call):
            need_to_be_remove = None

            for keyword in node.keywords:
                if keyword.arg == 'locations':
                    for sub_node in ast.walk(keyword):

                        # if the location path is exist, then turn to pack scope  # noqa: E501
                        if isinstance(
                                sub_node,
                                ast.Constant) and 'pack' in sub_node.value:
                            path = sub_node.value
                            if not osp.exists(
                                    osp.join(export_root_dir, path).replace(
                                        '.', '/')):
                                print(
                                    f"remove {osp.join(export_root_dir,path).replace('.','/')}"  # noqa: E501
                                )
                                need_to_be_remove = keyword
                                break

                if need_to_be_remove is not None:
                    break

            if need_to_be_remove is not None:
                node.keywords.remove(need_to_be_remove)

    with open(
            osp.join(export_module_dir, 'registry.py'), 'w',
            encoding='utf-8') as f:
        f.write(format_code(ast.unparse(ast_tree)))

    # postprocess for ImportFrom Node, turn to import from export path
    all_export_files = get_all_files(export_module_dir)
    for file in all_export_files:
        _transfer_to_export_import(file)

    # TODO: get demo.py
