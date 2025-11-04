# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import backoff
import openai
import os
import random
import time
from typing import List, Dict, Union
import re
import json
import httpx
import logging
try:
    from .abstract_language_model import AbstractLanguageModel
except:
    from abstract_language_model import AbstractLanguageModel

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,openai.InternalServerError,openai.APITimeoutError),
):
    """Retry a function with exponential backoff."""
    my_errors=tuple([openai.InternalServerError,openai.APITimeoutError,openai.RateLimitError])
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1
                logging.info(f"Error: {e}")
                # Check if max retries has been reached
                if num_retries > max_retries:
                    
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

           
            except Exception as e:
                raise(e)
            
    return wrapper




class GPT(AbstractLanguageModel):
    """
    The ChatGPT class handles interactions with the OpenAI models using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "chatgpt", cache: bool = False, retry_stra = True
    ) -> None:
        """
        Initialize the ChatGPT instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'chatgpt'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.retry = retry_stra
        if config_path=="":
            config_path=os.path.dirname(__file__)+"/config.json"
        self.config: Dict = self.config[model_name]

        self.model_id: str = self.config["model_id"]
    
        self.temperature: float = self.config["temperature"]

        self.max_tokens: int = self.config["max_tokens"]

        self.stop: Union[str, List[str]] = self.config["stop"]

        self.api_key: str = self.config["api_key"]
        if self.api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")
        self.api_baseurl=self.config["base_url"]


        if self.api_baseurl != "":
            self.client=openai.OpenAI(api_key=self.api_key,base_url=self.api_baseurl,timeout=100,max_retries=10)
        else:
            self.client=openai.OpenAI(api_key=self.api_key,timeout=100,max_retries=10)
        if self.cache and self.cache_log_file:
            self.respone_cache=json.load(open(self.cache_log_file,"r"))
    
    @retry_with_exponential_backoff
    def completions_with_backoff(self,**kwargs):
        try:    
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            self.logger.error(f"{e}")
            raise(e)
    def query(self, query: str,json_type=False,not_use_cache=False) -> str:
        if self.cache and not not_use_cache and query in self.respone_cache:
            return self.respone_cache[query]
        self.logger.info(
            f"Query: {query}"
        )

        response = self.chat([{"role": "user", "content": query}],json_type)

        
        result=self.get_response_text(response)
    
        self.logger.info(
            f"{self.model_id} Response: {result}"
        )
        if self.cache:
            with self.cache_lock:
                self.respone_cache[query] = result
                self.save_cache_log()
        assert isinstance(result,str)
        return result
    @backoff.on_exception(
        backoff.expo, openai.OpenAIError, max_time=20, max_tries=10
    )
    def chat(self, messages: List[Dict],json_type, num_responses: int = 1) -> Dict:
        """
        Send chat messages to the OpenAI model and retrieves the model's response.
        Implements backoff on OpenAI error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The OpenAI model's response.
        :rtype: Dict
        """
        kwargs={
            "model":self.model_id,
            "messages":messages,
            "temperature":self.temperature,
            "n":num_responses,
            "stop":self.stop,
            #"max_tokens":self.max_tokens,
        }
        if json_type:
            kwargs["response_format"]={"type":"json_object"}
        
        if self.retry:
            response = self.completions_with_backoff(**kwargs)
        else:
            response = self.client.chat.completions.create(**kwargs)
        

        return response

    def get_response_text(self, response:  Dict) -> str:

        if not isinstance(response.choices,List):
            result=response.choices.message.content
        else:
            result=response.choices[0].message.content
        try:
            reasoning_content = response.choices[0].message.reasoning_content
            result=f"<think> {reasoning_content} </think> {result}"
        except:
            pass
        return result
