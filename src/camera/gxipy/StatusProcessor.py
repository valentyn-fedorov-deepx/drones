#!/usr/bin/python
# -*- coding:utf-8 -*-
# -*-mode:python ; tab-width:4 -*- ex:set tabstop=4 shiftwidth=4 expandtab: -*-

import numpy
from loguru import logger
from .gxwrapper import *
from .dxwrapper import *
from .gxidef import *
from .gxiapi import *
from .Exception import *
import types

ERROR_SIZE = 1024

class StatusProcessor:
    def __init__(self):
        pass

    @staticmethod
    def process(status, class_name, function_name):
        """
        :brief      1.Error code processing
                    2.combine the class name and function name of the transmitted function into a string
                    3.Throw an exception
        :param      status:   function return value
        :param      class_name:  class name
        :param      function_name: function name
        :return:    none
        """
        if status != GxStatusList.SUCCESS:
            ret, err_code, string = gx_get_last_error(ERROR_SIZE)
            error_message = "%s.%s:%s" % (class_name, function_name, string)
            exception_deal(status, error_message)

    @staticmethod
    def printing(status, class_name, function_name):
        """
        :brief      1.Error code processing
                    2.combine the class name and function name of the transmitted function into a string and print it out
        :param      status:   function return value
        :param      class_name:  class name
        :param      function_name: function name
        :return:    none
        """
        if status != GxStatusList.SUCCESS:
            ret, err_code, string = gx_get_last_error(ERROR_SIZE)
            error_message = "%s.%s:%s" % (class_name, function_name, string)
            logger.debug(error_message)

