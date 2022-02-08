"""SOME-TOOLS errors module

This module contains custom error message for the SOME-TOOLS project.

"""


class Error(Exception):
    """ Base class for other exceptions """
    pass


class BadConfigurationFile(Error):
    """ Raised when important configuration check is not respected """
    pass


class MissingAttribute(Error):
    """ Raised when a mandatory class attribute is missing """
    pass


class FilesNotExisting(Error):
    """ Raised when a file-path is missing """
    pass
