import discord
from discord.ext import commands as bot

import numpy as np
import quantecon as qe
import time

class Nash(bot.Cog):

  def __init__(self, bot):
    self.bot = bot


  @bot.command()
  async def trust(self, ctx):
    mentions = ctx.message.mentions
    users = {ctx.author.id: 
      {
        "user": self.bot.get_user(ctx.author.id),
        "coins": 15
      }
    }

    for mention in mentions:
      user = self.bot.get_user(mention.id)
      if not user.bot:
        users[mention.id] = {
          "user": user,
          "coins": 15
        }

    if len(users) != 2:
      return await ctx.send("Requires exactly 2 people to play. Other modes not built.")

    """
    give give: -2+4   +4-2
    hide give: 4   -2
    give hide: -2     4
    hide hide: -2     -2
    """

    payoffs = np.array([[[2, 2], [4, -2]], [[-2, 4], [-2, -2]]])

    for u in users:
      user = users[u]["user"]
      starttext = "Welcome to the game of trust!\n\nThis game is inspired by https://ncase.me/trust/.\nHowever, this game allows you to compete against a real person!\n\n"
      starttext += "Here's how it works: You each start with **15** coins.\nEach round, you can choose to **cooperate** and give away **2** coins.\n"
      starttext += "If you do this, all your opponents get **4** coins. If one of your opponents does, you get **4** coins back.\n\n"
      starttext += "You can also choose to **cheat** and give nothing. If the other person doesn't cheat, you just get their **4** coins. But if they cheat too, you both get caught and both have to pay **2** coins anyway.\n\nThe first person to get to 30 coins wins! The game also exits if you are the last person standing. You get kicked out if you run out of coins.\n\n"
      await user.send(starttext + "\nThe payoff matrix:\n" + gameformat(payoffs, nash = False) + "\nLet's begin!")

    while True:
      for user in users:
        await users[user]["user"].send(f"Type one of **cooperate** or **cheat**, or type **exit** to quit. You have **{users[user]['coins']}** coins.")  

      actions = {}

      while True:
        if len(actions) == len(users):
          break

        message = await get(self.bot, users)
        if message == "timed out":
          for user in users:
            if user not in actions:
              actions[user] = False
              await users[user]["user"].send("Timed out. Going with **cheat**.")
        elif message.author.id not in actions:
          if message.content == "cheat":
            actions[message.author.id] = False
            await users[message.author.id]["user"].send("Going with **cheat**.  Waiting for opponents to finish...")

          elif message.content == "cooperate":
            actions[message.author.id] = True
            await users[message.author.id]["user"].send("Going with **cooperate**.  Waiting for opponents to finish...")

          elif message.content == "exit":
            for user in users:
              await users[user]["user"].send(f"{message.author} has quit the game.")
            del users[message.author.id]

      moves = []
      exit_now = False

      if not any(actions[u] for u in users):
        for u in users:
          users[u]["coins"] -= 2
          moves.append(f"{users[u]['user'].name} cheated. Everyone cheated, so they paid **2** coins and now have **{users[u]['coins']}** coins.")
          if users[u]["coins"] <= 0:
            moves.append(f"{users[u]['user'].name} has no coins left. They lost!")
          
      else:
        for u in users:
          if actions[u]:
            for v in users:
              if u != v:
                users[v]["coins"] += 4
                if users[v]["coins"] >= 30:
                  exit_now = True

            users[u]["coins"] -= 2

        for u in users:
          if actions[u]:
            moves.append(f"{users[u]['user'].name} cooperated, so they paid **2** coins and everyone else received **4** coins. They now have **{users[u]['coins']}** coins.")
          else:
            moves.append(f"{users[u]['user'].name} cheated. Accounting from coins from other people, they now have **{users[u]['coins']}** coins.")
          if users[u]["coins"] <= 0:
            moves.append(f"{users[u]['user'].name} has no coins left. They lost!")

      for user in users.copy():
        await users[user]["user"].send("Round finished! Here are the results:\n" +"\n".join(moves))
        if users[user]["coins"] <= 0:
          await users[user]["user"].send("Game over for you! Thanks for playing.")
          del users[user]

      if exit_now:
        for user in users:
          if users[user]["coins"] >= 30:
            await users[user]["user"].send("Game over! You won! Thanks for playing.")
          else:
            await users[user]["user"].send("Game over for you! Thanks for playing.")
        return

      if len(users) <= 1:
        for user in users:
          await users[user]["user"].send("Game over! You won! Thanks for playing.")
        return 
          





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
      func = "original QuantEcon pure_nash_brute"

    elif mode == "eliminate":
      a = time.time()
      results = eliminate_dominated(game)
      b = time.time()
      func = "dominated strategy elimination"

    elif mode == "compare":
      a = time.time()
      results = qe.game_theory.pure_nash.pure_nash_brute(game)
      b = time.time()
      results2 = eliminate_dominated(game)
      c = time.time()
      return await ctx.send(gameformat(game.payoff_profile_array, results) + "\n" + gameformat(game.payoff_profile_array, results2) + f"\nUsing `original QuantEcon pure_nash_brute` function ({round(b-a, 6)} s)\nvs `dominated strategy elimination` function ({round(c-b, 6)} s).")

    else:
      return await ctx.send("Please specify a valid solver mode.")
    
    await ctx.send(gameformat(game.payoff_profile_array, results) + f"\nUsing `{func}` function. Executed in {round(b-a, 6)} seconds.")


def eliminate_dominated(game):
  remaining = dominated_strategy_iterate(list(game.payoff_arrays), np.ones((2, 2)), 0, True, 0)
  if np.all(remaining): return []

  return [tuple(a) for a in np.argwhere(remaining)]

def dominated_strategy_iterate(payoffs, remaining, successful_iter, player, rounds_done):
  """
  simple recursive iterator for a 2x2 matrix game's pure Nash equilibria
  """
  # later: https://gambitproject.readthedocs.io/en/v16.0.1/pyapi.html
  # see: action deletion system
  if successful_iter == 2: 
    return remaining

  try:
    if np.all(payoffs[player][0] >= payoffs[player][1]):
      # zero is better
      payoffs[player] = np.delete(payoffs[player], 1, 0)
      payoffs[1 - player] = np.delete(payoffs[1 - player], 1, 1)
      if player:
        remaining[:, 1] = 0
      else:
        remaining[1, :] = 0
      successful_iter += 1
      rounds_done += 1

      return dominated_strategy_iterate(payoffs, remaining, successful_iter, 1 - player, rounds_done)

    elif np.all(payoffs[player][0] <= payoffs[player][1]):
      # one is better
      payoffs[player] = np.delete(payoffs[player], 0, 0)
      payoffs[1 - player] = np.delete(payoffs[1 - player], 0, 1)
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
      if rounds_done == 3:
        return remaining # return whatever shows up

      return dominated_strategy_iterate(payoffs, remaining, successful_iter, 1 - player, rounds_done)

  except:
    return remaining


def gameformat(game, results = [], nash = True):
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
  t = f">      ╚═════════╩═════════╝```"
  if nash: t += f"Pure Nash Equilibria: {','.join([str(game[result]) for result in results])}"
  text.append(t)
  return "\n".join(text)


async def get(bot, users):
  try:
    message = await bot.wait_for("message", timeout = 30, check = lambda m: m.author.id in users)
  except:
    return "timed out"

  return message


def setup(bot):
  bot.add_cog(Nash(bot))
