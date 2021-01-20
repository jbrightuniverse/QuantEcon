import discord
import random


def gameformat(game, results = [], nash = True, params = ["Me", "You", "C", "D"]):
  toprow = game[0]
  bottomrow = game[1]
  ul = toprow[0]
  ur = toprow[1]
  bl = bottomrow[0]
  br = bottomrow[1]
  include = []
  for result in results:
    include.append(result)

  text = ["> ```"]
  text.append(f">               {params[1]}")
  text.append(f">           {params[2]}         {params[3]}")
  text.append(">      ╔═════════╦═════════╗")
  text.append(f">    {params[2]} ║" + [" ", "*"][(0, 0) in include] + str(ul[0]).rjust(3)+","+str(ul[1]).ljust(4) + "│" + str(ur[0]).rjust(4)+","+str(ur[1]).ljust(3) + [" ", "*"][(0, 1) in include] + "║")
  text.append(f"> {params[0]}   ╠─────────┼─────────╣")
  text.append(f">    {params[3]} ║" + [" ", "*"][(1, 0) in include] + str(bl[0]).rjust(3)+","+str(bl[1]).ljust(4) + "│" + str(br[0]).rjust(4)+","+str(br[1]).ljust(3) + [" ", "*"][(1, 1) in include] + "║")
  t = f">      ╚═════════╩═════════╝```"
  if nash: t += f"Pure Nash Equilibria: {','.join([str(game[result]) for result in results])}"
  text.append(t)
  return "\n".join(text)


async def mbed(ctx, upper, lower, fields = [], thumbnail = None, footer = None):
  embed = discord.Embed(title = upper, description = lower, color = random.randint(0, 0xffffff))
  for field in fields:
    embed.add_field(name = field[0], value = field[1], inline = False)
  if thumbnail:
    embed.set_thumbnail(url=thumbnail)
  if footer:
    embed.set_footer(text = footer)
  await ctx.send(embed = embed)


async def get(bot, users):
  try:
    message = await bot.wait_for("message", timeout = 30, check = lambda m: m.author.id in users)
  except:
    return "timed out"

  return message