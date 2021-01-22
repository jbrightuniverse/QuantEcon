import discord
from discord.ext import commands as bot

import numpy as np
import quantecon as qe
from sympy.solvers import solve
from sympy import symbols, latex, lambdify, sympify, binomial, diff, simplify, re

from cogs.anarchy import tex

class Symbolic(bot.Cog):
  """:1234: Symbolic Math"""

  def __init__(self, bot):
    self.bot = bot

  @bot.command()
  async def solve(self, ctx):
    """
    Equation solver. Use this to solve or simplify equations.

    **Usage:**
    `-solve <expression>, [variable]`
    When specifying a variable to solve for, the expression in `<expression>` is set to equal zero

    **Examples:**
    `-solve 1+1`    2
    `-solve 3*x, x` 0
    """

    keys = ctx.message.content[7:].replace("`", "").split(",")
    async with ctx.typing():
      equation = sympify(keys[0])
      if len(keys) == 2:
        equation = solve(equation, symbols(keys[1].lstrip().rstrip()))
      await tex(ctx, str(latex(equation)), "Result:")

def setup(bot):
  bot.add_cog(Symbolic(bot))