from spade.template import Template

PERFORMATIVE = "performative"
REQUEST = "request"
INFORM = "inform"
ACKNOWLEDGE = "acknowledge"

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