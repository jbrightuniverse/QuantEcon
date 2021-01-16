import discord
from discord.ext import commands as bot

import numpy as np
import quantecon as qe

class DP(bot.Cog):

  def __init__(self, bot):
    self.bot = bot

  @bot.command()
  async def dpsolve(self, ctx):
    keywords = ctx.message.content[9:].lower().replace(" ", "")
    beta = 0.5
    reward = 1
    sets = keywords.split(",")
    for entry in sets:
      kvpair = entry.split("=")
      if kvpair[0] == "beta":
        beta = float(kvpair[1])
      elif kvpair[0] == "y":
        reward = float(kvpair[1])
    ddp = qe.markov.ddp.DiscreteDP(np.array([reward]), np.array([[1]]), beta, [1], [1])
    res = ddp.solve(method='value_iteration', v_init=[0])
    await ctx.send(f"Optimal Value Function: {res.v}\nOptimal Policy Function: {res.sigma}\nCompleted in {res.num_iter} iterations using original QuantEcon {res.method} method.")




def setup(bot):
  bot.add_cog(DP(bot))