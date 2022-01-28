"""SOME-TOOLS errors module

This module contains custom error message for the SOME-TOOLS project.

"""


class Error(Exception):
    """ Base class for other exceptions """
    pass


class BadConfigurationFile(Error):
    """ Raised when important configuration check is not respected """
    pass
