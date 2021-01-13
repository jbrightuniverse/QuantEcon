import discord
from discord.ext import commands as bot

class DP(bot.Cog):

  def __init__(self, bot):
    self.bot = bot

  @bot.command()
  async def hello(self, ctx):
    await ctx.send("hi")




def setup(bot):
  bot.add_cog(DP(bot))