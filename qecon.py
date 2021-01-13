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


@bot.command(name="reboot")
async def rl(ctx, ext):
  try:
    bot.reload_extension(f"cogs.{ext}")
    await ctx.send(f"reloaded {ext} extension")
  except:
    bot.load_extension(f"cogs.{ext}")
    await ctx.send(f"loaded {ext} extension")
  

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

bot.run(TOKEN, bot=True, reconnect=True)