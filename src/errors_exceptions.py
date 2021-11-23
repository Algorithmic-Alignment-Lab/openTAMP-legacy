class OpenTAMPException(Exception):
    """ A generic exception for OpenTAMP """

class ProblemConfigException(OpenTAMPException):
    """ Either config not found or config format incorrect """
    pass

class DomainConfigException(OpenTAMPException):
    """ Either config not found or config format incorrect """
    pass

class SolversConfigException(OpenTAMPException):
    """ Either config not found or config format incorrect """
    pass

class ParamValidationException(OpenTAMPException):
    """ Check validate_params functions """
    pass

class PredicateException(OpenTAMPException):
    """ Predicate type mismatch, not defined, or parameter range violation """
    pass

class HLException(OpenTAMPException):
    """ An issue with the high level solver (hl_solver) """
    pass

class LLException(OpenTAMPException):
    """ An issue with the low level solver (ll_solver) """
    pass

class OpenRAVEException(OpenTAMPException):
    """ An OpenRAVE related issue"""
    pass
class ImpossibleException(OpenTAMPException):

    """ This exception should never be raised """
    pass
