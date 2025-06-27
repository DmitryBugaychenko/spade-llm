from typing import Any, Dict, Optional, Callable, List
import logging
from spade_llm.core.conf import ConfigurableRecord, Configurable, configuration
from pydantic import BaseModel
from pydantic.fields import Field
from spade_llm.core.tools import ToolFactory
from langchain_core.tools import tool
from math import sin, cos, tan, radians, degrees, asin, acos, atan
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools.base import BaseToolkit

logger = logging.getLogger(__name__)

class TrigonometryCalculatorConf(ConfigurableRecord):
    precision: int = Field(default=4, description="The number of decimal places to round the trigonometric value to")


class TrigonometryToolkit(BaseToolkit):
    precision: int = Field(default=4, description="The number of decimal places to round the trigonometric value to")

    def __init__(self, precision: int):
        super().__init__()
        self.precision = precision

    @property
    def get_tools(self) -> List[BaseTool]:
        """Возвращает список инструментов (все тригонометрические функции)."""
        lst = [
            self._create_sin_tool(),
            self._create_cos_tool(),
            self._create_tan_tool(),
            self._create_cot_tool()
        ]
        return lst

    def _create_sin_tool(self) -> BaseTool:
        precision = self.precision

        @tool
        def sin_calculator(
                angle: float,
                angle_unit: str = "degrees",
        ) -> float:
            """
            Выполняет тригонометрический расчет синуса.

            Аргументы:
                angle: Угол для вычисления
                angle_unit: Единицы измерения угла ('degrees' или 'radians')

            Возвращает:
                Результат вычисления в виде float

            Примеры:
                >>> sin_calculator(30)  # sin(30°)
                >>> sin_calculator(0.5, angle_unit="radians")  # sin(0.5 rad)
            """
            logger.info("Calculating sin with {} digits precision".format(precision))
            if angle_unit == "degrees":
                angle_rad = radians(angle)
            else:
                angle_rad = angle

            result = sin(angle_rad)
            return round(float(result), precision)

        return sin_calculator

    def _create_cos_tool(self) -> BaseTool:
        precision = self.precision

        @tool
        def cos_calculator(
                angle: float,
                angle_unit: str = "degrees",
        ) -> float:
            """
            Выполняет тригонометрический расчет косинуса.

            Аргументы:
                angle: Угол для вычисления
                angle_unit: Единицы измерения угла ('degrees' или 'radians')

            Возвращает:
                Результат вычисления в виде float

            Примеры:
                >>> cos_calculator(60)  # cos(60°)
                >>> cos_calculator(1.047, angle_unit="radians")  # cos(1.047 rad)
            """
            if angle_unit == "degrees":
                angle_rad = radians(angle)
            else:
                angle_rad = angle

            result = cos(angle_rad)
            return round(float(result), precision)

        return cos_calculator

    def _create_tan_tool(self) -> BaseTool:
        precision = self.precision

        @tool
        def tan_calculator(
                angle: float,
                angle_unit: str = "degrees",
        ) -> float:
            """
            Выполняет тригонометрический расчет тангенса.

            Аргументы:
                angle: Угол для вычисления
                angle_unit: Единицы измерения угла ('degrees' или 'radians')

            Возвращает:
                Результат вычисления в виде float

            Примеры:
                >>> tan_calculator(45)  # tan(45°)
                >>> tan_calculator(0.785, angle_unit="radians")  # tan(0.785 rad)
            """
            if angle_unit == "degrees":
                angle_rad = radians(angle)
            else:
                angle_rad = angle

            result = tan(angle_rad)
            return round(float(result), precision)

        return tan_calculator

    def _create_cot_tool(self) -> BaseTool:
        precision = self.precision

        @tool
        def cot_calculator(
                angle: float,
                angle_unit: str = "degrees",
        ) -> float:
            """
            Выполняет тригонометрический расчет котангенса.

            Аргументы:
                angle: Угол для вычисления
                angle_unit: Единицы измерения угла ('degrees' или 'radians')

            Возвращает:
                Результат вычисления в виде float

            Примеры:
                >>> cot_calculator(45)  # cot(45°)
                >>> cot_calculator(0.785, angle_unit="radians")  # cot(0.785 rad)
            """
            if angle_unit == "degrees":
                angle_rad = radians(angle)
            else:
                angle_rad = angle

            result = 1 / tan(angle_rad)
            return round(float(result), precision)

        return cot_calculator



@configuration(TrigonometryCalculatorConf)
class TrigonometryCalculatorToolFactory(ToolFactory, Configurable[TrigonometryCalculatorConf]):
    def create_tool(self) -> List[BaseTool]:
        return self.config.create_kwargs_instance(TrigonometryToolkit).get_tools

