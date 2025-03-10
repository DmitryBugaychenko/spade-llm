from re import template

from spade.template import Template

PERFORMATIVE = "performative"
REQUEST = "request"
REQUEST_PROPOSAL = "request_proposal"
INFORM = "inform"
PROPOSE = "propose"
ACCEPT = "accept"
REFUSE = "refuse"
ACKNOWLEDGE = "acknowledge"
FAILURE = "failure"

class Templates:
    @staticmethod
    def REQUEST() -> Template:
        return Template(metadata={PERFORMATIVE : REQUEST})

    @staticmethod
    def REQUEST_PROPOSAL() -> Template:
        return Template(metadata={PERFORMATIVE : REQUEST_PROPOSAL})

    @staticmethod
    def INFORM() -> Template:
        return Template(metadata={PERFORMATIVE : INFORM})

    @staticmethod
    def ACKNOWLEDGE() -> Template:
        return Template(metadata={PERFORMATIVE : ACKNOWLEDGE})

    @staticmethod
    def FAILURE() -> Template:
        return Template(metadata={PERFORMATIVE : FAILURE})

    @staticmethod
    def PROPOSE() -> Template:
        return Template(metadata={PERFORMATIVE : PROPOSE})

    @staticmethod
    def ACCEPT() -> Template:
        return Template(metadata={PERFORMATIVE : ACCEPT})

    @staticmethod
    def REFUSE() -> Template:
        return Template(metadata={PERFORMATIVE : REFUSE})

    @classmethod
    def from_thread(cls, thread):
        return Template(thread=thread)