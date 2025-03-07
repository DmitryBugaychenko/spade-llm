import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import StructuredTool
from langchain_gigachat.chat_models import GigaChat
from openapi_parser import parse
from openapi_parser.enumeration import DataType
from pydantic import create_model, Field
from pydantic.fields import FieldInfo


def to_type(t : DataType) -> type:
    """Converts OpenAPI datatype to python type"""
    match t:
        case DataType.INTEGER:
            return int
        case DataType.NUMBER:
            return int
        case DataType.BOOLEAN:
            return bool
        case DataType.STRING:
            return str
    return str

def dummy(arg: Any) -> Any :
    """Just a placeholder to use StructuredTool for creating tool description"""
    return None

def check_flag(flag: str, extensions: dict) -> bool:
    """Checks whenever a flag is set in a dictionary. Useful for checking extensions flags."""
    return flag in extensions.keys() and extensions[flag]

def construct_field(field) -> Field:
    if field.required:
        return Field(description=field.description)
    else:
        if field.schema.default:
            return Field(description=field.description, default=field.schema.default)
        else:
            return Field(description=field.description, default="default")

def construct_tool(operation) -> StructuredTool:
    """Parses definition of the OpenAPI operation and create matching tool description"""
    args: dict[str, (type, FieldInfo)] = dict()
    for arg in operation.parameters:
        if not check_flag("context", arg.extensions):
            field = construct_field(arg)
            args[arg.name] = (to_type(arg.schema.type), field)
    arg_schema = create_model("Arguments", **args)

    return StructuredTool.from_function(
        func=dummy,
        description=operation.description,
        name=operation.summary,
        args_schema=arg_schema,
        infer_schema=False,
        parse_docstring=False
    )

class bcolors:
    """Codes to highlight console"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def fail(test: str, reason: str):
    print(f"'{test}' {bcolors.FAIL}FAILED{bcolors.ENDC}: {reason}")

def success(test: str):
    print(f"'{test}' {bcolors.OKGREEN}PASSED{bcolors.ENDC}")

def main():

    # Read schema from file
    file = sys.argv[1]
    print("Reading file " + file)
    content = Path(file).read_text()
    schema = parse(spec_string=content)

    # Configura models for testing in form (name, model, number of test runs)
    models : list[(str, GigaChat, int)] = [
        ("GigaChat-lite", GigaChat(
            credentials=os.environ['GIGA_CRED'],
            model="GigaChat",
            verify_ssl_certs=False),
         5),
        ("GigaChat-pro", GigaChat(
            credentials=os.environ['GIGA_CRED'],
            model="GigaChat-Pro",
            verify_ssl_certs=False),
         3),
        ("GigaChat-max", GigaChat(
            credentials=os.environ['GIGA_CRED'],
            model="GigaChat-Pro",
            verify_ssl_certs=False),
         3)
    ]

    # Configure basic system prompt hinting to use tools
    system_prompt = SystemMessage("You are an AI agent. Use provided tools to solve answer user questions. "
                                  "If not proper tool found answer 'Not tool available'")

    for path in schema.paths:
        for operation in path.operations:

            # Skip non AI-ready API
            if not check_flag("AI_ready", operation.extensions):
                print("Skipping non-AI ready '{0}'".format(operation.summary))
                continue

            # Make sure examples for testing are provided
            if "examples" not in operation.extensions:
                fail(operation.summary, "no examples provided")
                continue

            tool = construct_tool(operation)

            for name, model, rounds in models:
                print("Testing '{0}' for {1}".format(operation.summary, name))
                successes = 0.0
                attempts = 0.0

                for round in range(1,rounds+1):
                    print(f"Round {round}")
                    giga: GigaChat = model
                    with_tools = giga.bind_tools([tool])

                    for example in operation.extensions["examples"].values():
                        prompt = example["prompt"]
                        attempts += 1
                        response : AIMessage = with_tools.invoke([system_prompt, prompt])
                        if len(response.tool_calls) == 1:
                            call = response.tool_calls[0]

                            from_call = tool.args_schema.model_validate(call["args"])
                            expected = tool.args_schema.model_validate(example["args"] if example["args"] else dict())

                            if from_call == expected:
                                successes += 1
                                success(prompt)
                            else:
                                fail(prompt, f"Expected args {expected}, provided args {from_call}")
                        else:
                            fail(prompt, "no tool call generated")

                print(f"Precision for '{operation.summary}' with {name} is {successes / attempts * 100}%")

if __name__ == "__main__":
    main()