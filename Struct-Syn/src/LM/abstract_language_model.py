# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
import json
import os
import logging
import re
from threading import Lock
class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "", cache: bool = False
    ) -> None:
        """
        Initialize the AbstractLanguageModel instance with configuration, model details, and caching options.

        :param config_path: Path to the config file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the language model. Defaults to "".
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict = None
        self.model_name: str = model_name
        self.cache = cache
        if self.cache:
            self.cache_lock=Lock()
            self.respone_cache: Dict[str, List[Any]] = {}
            self.cache_log_file=None
        self.load_config(config_path)
    def load_config(self, path: str) -> None:
        """
        Load configuration from a specified path.

        :param path: Path to the config file. If an empty path provided,
                     default is `config.json` in the current directory.
        :type path: str
        """
        if path == "":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "config.json")

        with open(path, "r") as f:
            self.config = json.load(f)

        self.logger.debug(f"Loaded config from {path} for {self.model_name}")

    def clear_cache(self) -> None:
        """
        Clear the response cache.
        """
        self.respone_cache.clear()

    @abstractmethod
    def query(self, query: str,not_use_cache:bool=False) -> Any:
        """
        Abstract method to query the language model.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_response_text(self, query_responses):
        """
        Abstract method to extract response texts from the language model's response(s).

        :param query_responses: The responses returned from the language model.
        :type query_responses: Union[List[Dict], Dict]
        :return: List of textual responses.
        :rtype: List[str]
        """
        pass
    def set_cache_log(self,cache_log_file:str):
        if self.cache:
            with self.cache_lock:
                if  os.path.exists(cache_log_file):
                    self.respone_cache=json.load(open(cache_log_file,"r"))
            self.cache_log_file=cache_log_file
    def save_cache_log(self):
        if self.cache and self.cache_log_file and self.respone_cache:
            json.dump(self.respone_cache,open(self.cache_log_file,"w"),indent=4,ensure_ascii=False)

    def remove_markdown(self,text):
        # 移除标题
        text = re.sub(r'#+\s*', '', text)
        # 移除粗体和斜体
        text = re.sub(r'\*\*|\*|__|', '', text)
        # 移除链接和图片
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'!\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # 移除代码块和内联代码
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'`.*?`', '', text)
        # 移除列表标记
        text = re.sub(r'[\*\+\-]\s+', '', text)
        # 移除引用
        text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)
        # 移除水平线
        text = re.sub(r'---', '', text)
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        return text

