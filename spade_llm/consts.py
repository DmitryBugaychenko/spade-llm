from re import template

from spade.template import Template

PERFORMATIVE = "performative"
REQUEST = "request"
INFORM = "inform"
ACKNOWLEDGE = "acknowledge"
FAILURE = "failure"

class Templates:
    @staticmethod
    def REQUEST() -> Template:
        return Template(metadata={PERFORMATIVE : REQUEST})

    @staticmethod
    def INFORM() -> Template:
        return Template(metadata={PERFORMATIVE : INFORM})

    @staticmethod
    def ACKNOWLEDGE() -> Template:
        return Template(metadata={PERFORMATIVE : ACKNOWLEDGE})

    @staticmethod
    def FAILURE() -> Template:
        return Template(metadata={PERFORMATIVE : FAILURE})

    @classmethod
    def from_thread(cls, thread):
        return Template(thread=thread)