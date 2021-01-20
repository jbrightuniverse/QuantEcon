import discord
from discord.ext import commands as bot

import numpy as np
import quantecon as qe

class DP(bot.Cog):
  """:timer: Dynamic Programming"""

  def __init__(self, bot):
    self.bot = bot

  @bot.command()
  async def dpsolve(self, ctx):
    """
    Solves a basic dynamic program given beta discount factor and an initial y assuming no growth in y.

    **Usage:**
    `-dpsolve beta = <beta>, y = <y>`

    **Examples:**
    `-dpsolve beta = 0.5, y = 1` solves a dynamic program with `beta = 0.5` and `y = 1`
    `-dpsolve y = 1, beta = 0.5` same as the above
    """

    keywords = ctx.message.content[9:].lower().replace(" ", "")
    beta = 0.5
    y = 1
    sets = keywords.split(",")
    for entry in sets:
      kvpair = entry.split("=")
      if kvpair[0] == "beta":
        beta = float(kvpair[1])
      elif kvpair[0] == "y":
        y = float(kvpair[1])
    ddp = qe.markov.ddp.DiscreteDP(np.array([y, W]), np.array([[1]]), beta, [1], [1])
    res = ddp.solve(method='value_iteration', v_init=[0])
    await ctx.send(f"Optimal Value Function: {res.v}\nOptimal Policy Function: {res.sigma}\nCompleted in {res.num_iter} iterations using original QuantEcon {res.method} method.")




def setup(bot):
  bot.add_cog(DP(bot))