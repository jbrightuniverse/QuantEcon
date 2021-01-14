import discord
from discord.ext import commands as bot

import numpy as np
import quantecon as qe
import time

class Nash(bot.Cog):

  def __init__(self, bot):
    self.bot = bot

  @bot.command()
  async def randmatx(self, ctx):
    actions = ctx.message.content[10:]

    if not actions or not all(a.lstrip().rstrip().isdigit() for a in actions.split(",")):
      return await ctx.send("Please provide at least one **integer** number of actions.")

    try: 
      rho = int(actions.split("cov")[1])
      if rho > 1 or rho < -1/(rho-1): raise Exception()
    except:
      rho = 0

    actions = actions.split("cov")[0]
    game = qe.game_theory.random.covariance_game([int(a) for a in actions.split(",")], rho)
    await ctx.send(game)


  @bot.command()
  async def randgame(self, ctx):
    game = qe.game_theory.random.covariance_game((2, 2), 0).payoff_profile_array * 10
    game = game.astype(np.int8)
    await ctx.send(gameformat(game))


  @bot.command()
  async def randsolve(self, ctx, mode = "brute"):
    profile = qe.game_theory.random.covariance_game((2, 2), 0).payoff_profile_array * 10
    game = qe.game_theory.normal_form_game.NormalFormGame(profile.astype(np.int8), dtype = np.int8)

    results = []
    func = None
    a = 0
    b = 0

    if mode == "brute":
      a = time.time()
      results = qe.game_theory.pure_nash.pure_nash_brute(game)
      b = time.time()
      await ctx.send(game.payoff_arrays)
      await ctx.send(game.payoff_profile_array)
      func = "original QuantEcon pure_nash_brute"

    elif mode == "eliminate":
      a = time.time()
      results = await eliminate_dominated(ctx, game)
      b = time.time()
      await ctx.send(results)
      func = "dominated strategy elimination"
    
    else:
      return await ctx.send("Please specify a valid solver mode.")
    
    await ctx.send(gameformat(game.payoff_profile_array, results) + f"\nUsing `{func}` function. Executed in {round(b-a, 6)} seconds.")


async def eliminate_dominated(ctx, game):
  remaining = dominated_strategy_iterate(list(game.payoff_arrays), np.ones((2, 2)), 0, True, 0)
  await ctx.send(remaining)
  if np.all(remaining): return []

  return tuple(map(tuple, np.where(remaining)))

def dominated_strategy_iterate(payoffs, remaining, successful_iter, player, rounds_done):
  """
  simple recursive iterator for a 2x2 matrix game's pure Nash equilibria
  """

  if successful_iter == 2: 
    return remaining

  if np.all(payoffs[player][0] > payoffs[player][1]):
    # zero is better
    payoffs[player] = np.delete(payoffs[player], 1, 0)
    if player:
      remaining[:, 1] = 0
    else:
      remaining[1, :] = 0
    successful_iter += 1
    rounds_done += 1

    return dominated_strategy_iterate(payoffs, remaining, successful_iter, 1 - player, rounds_done)

  elif np.all(payoffs[player][0] < payoffs[player][1]):
    # one is better
    payoffs[player] = np.delete(payoffs[player], 0, 0)
    if player:
      remaining[:, 0] = 0
    else:
      remaining[0, :] = 0
    successful_iter += 1
    rounds_done += 1
    return dominated_strategy_iterate(payoffs, remaining, successful_iter, 1 - player, rounds_done)
    

  else:
    # neither is better, try flipping
    rounds_done += 1
    if rounds_done == 2:
      return remaining # return whatever shows up

    return dominated_strategy_iterate(payoffs, remaining, successful_iter, 1 - player, rounds_done)


def gameformat(game, results = []):
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
  text.append(">               You")
  text.append(">           C         D")
  text.append(">      ╔═════════╦═════════╗")
  text.append(">    C ║" + [" ", "*"][(0, 0) in include] + str(ul[0]).rjust(3)+","+str(ul[1]).ljust(4) + "│" + str(ur[0]).rjust(4)+","+str(ur[1]).ljust(3) + [" ", "*"][(0, 1) in include] + "║")
  text.append("> Me   ╠─────────┼─────────╣")
  text.append(">    D ║" + [" ", "*"][(1, 0) in include] + str(bl[0]).rjust(3)+","+str(bl[1]).ljust(4) + "│" + str(br[0]).rjust(4)+","+str(br[1]).ljust(3) + [" ", "*"][(1, 1) in include] + "║")
  text.append(f">      ╚═════════╩═════════╝```Pure Nash Equilibria: {','.join([str(game[result]) for result in results])}")
  return "\n".join(text)


def setup(bot):
  bot.add_cog(Nash(bot))
