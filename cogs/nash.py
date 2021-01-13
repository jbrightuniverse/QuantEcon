import discord
from discord.ext import commands as bot

import quantecon.game_theory as gt

class Nash(bot.Cog):

  def __init__(self, bot):
    self.bot = bot

  
  @bot.command()
  async def randgame(self, ctx):
    actions = ctx.message.content[10:]

    if not actions or not all(a.lstrip().rstrip().isdigit() for a in actions.split(",")):
      return await ctx.send("Please provide at least one **integer** number of actions.")

    try: 
      rho = int(actions.split("cov")[1])
      if rho > 1 or rho < -1/(rho-1): raise Exception()
    except:
      rho = 0

    game = gt.random.covariance_game([int(a) for a in actions], rho)
    await ctx.send(game)




def setup(bot):
  bot.add_cog(Nash(bot))
