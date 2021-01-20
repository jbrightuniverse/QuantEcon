import util

import discord
from discord.ext import commands

import importlib
import os
import traceback
import sys
import json
import asyncio

from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TOKEN")
bot = commands.Bot(command_prefix="-", case_insensitive = True, intents = discord.Intents.all())
bot.remove_command("help")

for ext in os.listdir("cogs"):
  if ext.endswith(".py"):
    bot.load_extension("cogs."+ext[:-3])
    print(ext + " loaded")

@bot.event
async def on_ready():
  await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="-help"))
  print(f"Ready: {bot.user}")


@bot.command()
async def help(ctx, *args):
  cmdlist = []
  cogdict = defaultdict(list)
  for name in sorted(bot.cogs):
    cog = bot.get_cog(name)
    for cmdx in cog.walk_commands():
      cmd = bot.get_command(cmdx.qualified_name)
      if cmd.help:
        cogdict[name].append(f"**{cmdx.qualified_name}**")
        cmdlist.append(cmdx.qualified_name)
      cogdict[name].sort()
  if not args:
    fields = list(zip([bot.get_cog(name).description for name in cogdict], [", ".join(cogdict[name]) for name in cogdict]))
    fields.sort(key = lambda x: x[0].split()[1])
    fields.append(["Created by <@375445489627299851> and based on https://quanteconpy.readthedocs.io/en/latest/.\n\nThis bot is available on GitHub: https://github.com/jbrightuniverse/QuantEcon", f"v0.0.2"])
    return await util.mbed(ctx, "QuantEcon", "I'm a bot designed for running various experiments and algorithms related to economics.\n\nType **-help <command>** (e.g. `-help solve`) for more details.", fields = fields, thumbnail = bot.user.avatar_url)
  if args[0] not in cmdlist or not bot.get_command(args[0]).help:
    return await util.mbed(ctx, f"-{args[0]}", "Sorry, that command doesn't exist.", thumbnail = bot.user.avatar_url)
  return await util.mbed(ctx, f"-{args[0]}", bot.get_command(args[0]).help, thumbnail = bot.user.avatar_url)


@bot.event
async def on_command_error(ctx, error):
  if isinstance(error, commands.CommandNotFound):
    return
  elif isinstance(error, commands.MissingRequiredArgument):
    return
  elif isinstance(error, commands.CommandOnCooldown):
    await ctx.send(f"**{ctx.author}**, this command is on cooldown. Try again in {error.retry_after} seconds")
  else:
    msg = "An error has occurred!\n```" + "".join(traceback.format_exception(type(error), error, error.__traceback__, 999)) + "```"
    await ctx.send(msg[:2000])
    if len(msg) > 2000:
      try:
        print(msg)
      except:
        pass

bot.run(TOKEN, bot=True, reconnect=True)