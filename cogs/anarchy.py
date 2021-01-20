import discord
from discord.ext import commands as bot

import asyncio
import math
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
import random
import requests
import urllib.parse

from io import BytesIO
from PIL import Image
from sympy.plotting import plot
from sympy.solvers import solve
from sympy import symbols, latex, lambdify, sympify, binomial, diff, simplify, re
import sympy

from util import *

import warnings
warnings.filterwarnings("ignore")

class Anarchy(bot.Cog):
  """:desktop: Price of Anarchy Experiment"""

  def __init__(self, bot):
    self.bot = bot

  @bot.command()
  async def polynomial(self, ctx):
    """
    Computes the π probability values for the Price of Anarchy with 2 routers and n computers.
    
    This command uses Newton's method of polynomial root approximation to determine the solutions to the n-1th degree polynomial system of equations dictating the expected payoffs for each player in a mixed strategy.

    **Usage:**
    `-polynomial <num_players> [max | all]`

    **Examples:**
    `-polynomial 2`     Displays β vs π for Nash equilibrium π given 2 players
    `-polynomial 3 max` Displays β vs π for socially optimal π given 3 players
    `-polynomial 5 all` Gives Nash eq. and socially optimal π given 5 players
    """

    if ctx.message.content.endswith("max"):
      ctx.message.content = ctx.message.content[:-3]
      maximize = True
    elif ctx.message.content.endswith("all"):
      ctx.message.content = ctx.message.content[:-3]
      maximize = 2
    else:
      maximize = False

    async with ctx.typing():
      n = int(ctx.message.content[12:])
      if n > 50: return await ctx.send("too big")

      if not maximize: n -= 1

      pi, beta = symbols("pi beta")
      if maximize:
        equation = sum((-1)**(k+1) * pi**k * binomial(n-1, k) for k in range(0, n-1)) + (beta + (-1)**n)*pi**(n-1)
      else:
        equation = sum((beta - (-1)**k * binomial(n+1, k+1)) * pi**k for k in range(0, n+1))

      eqdiff = diff(equation, pi)

      guesses = []
      for beta in np.append(np.linspace([1/(n+1),0][maximize > 0], 1, [20, 40][maximize > 0])[:-1], 0.99999999):
        cur_guess = beta
        while True:
          await asyncio.sleep(0)
          new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
          
          if abs(new_cur_guess - cur_guess) <= 0.001:
            break

          cur_guess = new_cur_guess

        guesses.append(re(cur_guess))

      guesses2 = []
      if maximize == 2:
        nx = n - 1
        equation = sum((symbols("beta") - (-1)**k * binomial(nx+1, k+1)) * pi**k for k in range(0, nx+1))
        eqdiff = diff(equation, pi)
        for beta in np.append(np.linspace(1/(nx+1), 1, 20)[:-1], 0.99999999):
          cur_guess = beta
          while True:
            await asyncio.sleep(0)
            new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
            
            if abs(new_cur_guess - cur_guess) <= 0.001:
              break

            cur_guess = new_cur_guess

          guesses2.append(re(cur_guess))

      plt.clf()
      fig, ax = plt.subplots()
      space = np.append(np.linspace([1/(n+1),0][maximize > 0], 1, [20, 40][maximize > 0])[:-1], 0.99999999)
      if n == 1 + min(maximize, 1):
        beta = symbols("beta")
        if maximize:
          exp = 1/(1+beta)
          label = 'Symbolic Answer (Optimal)'
        else:
          exp = (2-beta)/(1+beta)
          label = 'Symbolic Answer (Nash)'

        mapper = lambdify(beta, exp, modules=['numpy'])
        xs = np.linspace(0, 1, 100)
        ys = mapper(xs)
        plt.plot(xs, ys, label=label)

        if maximize == 2:
          mapper = lambdify(beta, (2-beta)/(1+beta), modules=['numpy'])
          xs = np.linspace(0, 1, 100)
          ys = mapper(xs)
          plt.plot(xs, ys, label='Symbolic Answer (Nash)')

      elif n == 2 + min(maximize, 1):
        beta = symbols("beta")
        if maximize:
          exp = (sympy.sqrt(beta) - 1)/(beta - 1)
          label = 'Symbolic Answer (Optimal)'
        else:
          exp = ((3+beta) - sympy.sqrt(-3*beta**2 + 22*beta - 3)) / (2*(1-beta)) 
          label = 'Symbolic Answer (Nash)'

        mapper = lambdify(beta, exp, modules=['numpy'])
        xs = np.linspace(0, 1, 100)
        ys = mapper(xs)
        plt.plot(xs, ys, label=label)

        if maximize == 2:
          mapper = lambdify(beta, ((3+beta) - sympy.sqrt(-3*beta**2 + 22*beta - 3)) / (2*(1-beta)) , modules=['numpy'])
          xs = np.linspace(0, 1, 100)
          ys = mapper(xs)
          plt.plot(xs, ys, label='Symbolic Answer (Nash)')

      else:
        if maximize != 2:
          plt.plot(space, guesses, label = 'Direct Fit')
        else:
          space2 = np.append(np.linspace(1/(n), 1, 20)[:-1], 0.99999999)
          plt.plot(space, guesses, label = 'Direct Fit (Optimal)')
          plt.plot(space2, guesses2, label = 'Direct Fit (Nash)')

      if maximize == 1:
        title = f"β vs Socially Optimal Mixing Probability: n = {n} players"
      elif maximize == 2:
        title = f"β vs Optimal and Nash Mixing Probability: n = {n} players"
      else:
        title = f"β vs Mixed Nash Equilibrium Probability: n = {n+1} players"

      if maximize == 0:
        label = 'Newton\'s Method Appx. (Nash)'
      else:
        label = 'Newton\'s Method Appx. (Optimal)'
      plt.plot(space, guesses, "*", color = "green", label = label)
      if maximize == 2:
        space2 = np.append(np.linspace(1/(n), 1, 20)[:-1], 0.99999999)
        plt.plot(space2, guesses2, "*", color = "purple", label = 'Newton\'s Method Appx. (Nash)')

      fig.suptitle(title)
      ax.legend()
      plt.xlabel("β")
      plt.ylabel("π")
      ax.set_xlim([[1/(n+1),0][maximize > 0],1])
      ax.set_ylim([0.5,1])
      filex = BytesIO()
      fig.savefig(filex, format = "png")
      filex.seek(0)
      plt.close()
      await ctx.send(file=discord.File(filex, "nash.png"))

  @bot.command()
  async def polyall(self, ctx):
    """
    Computes the π probability values for the Price of Anarchy with 2 routers and n computers for n between 2 and 20.

    Takes the results and compiles them into a GIF.

    This command uses Newton's method of polynomial root approximation to determine the solutions to the n-1th degree polynomial system of equations dictating the expected payoffs for each player in a mixed strategy.

    **Usage:**
    `-polyall[max | all]`

    **Examples:**
    `-polyall`     Displays β vs π for Nash equilibrium π
    `-polyall max` Displays β vs π for socially optimal π
    `-polyall all` Gives Nash eq. and socially optimal π
    """

    if ctx.message.content.endswith("max"):
      maximize = True
    elif ctx.message.content.endswith(" all"):
      maximize = 2
    else:
      maximize = False

    msg = await ctx.send("Working...")
    async with ctx.typing():
      images = []
      for n in range(2, 21):
        if not maximize: n -= 1
        pi, beta = symbols("pi beta")
        if maximize:
          equation = sum((-1)**(k+1) * pi**k * binomial(n-1, k) for k in range(0, n-1)) + (beta + (-1)**n)*pi**(n-1)
        else:
          equation = sum((beta - (-1)**k * binomial(n+1, k+1)) * pi**k for k in range(0, n+1))

        eqdiff = diff(equation, pi)

        guesses = []
        for beta in np.append(np.linspace([1/(n+1),0][maximize > 0], 1, [20, 40][maximize > 0])[:-1], 0.99999999):
          cur_guess = beta
          while True:
            await asyncio.sleep(0)
            new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
            
            if abs(new_cur_guess - cur_guess) <= 0.001:
              break

            cur_guess = new_cur_guess

          guesses.append(re(cur_guess))

        guesses2 = []
        if maximize == 2:
          nx = n - 1
          equation = sum((symbols("beta") - (-1)**k * binomial(nx+1, k+1)) * pi**k for k in range(0, nx+1))
          eqdiff = diff(equation, pi)
          for beta in np.append(np.linspace(1/(nx+1), 1, 20)[:-1], 0.99999999):
            cur_guess = beta
            while True:
              await asyncio.sleep(0)
              new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
              
              if abs(new_cur_guess - cur_guess) <= 0.001:
                break

              cur_guess = new_cur_guess

            guesses2.append(re(cur_guess))

        plt.clf()
        fig, ax = plt.subplots()
        space = np.append(np.linspace([1/(n+1),0][maximize > 0], 1, [20, 40][maximize > 0])[:-1], 0.99999999)
        if maximize != 2:
          plt.plot(space, guesses, label = 'Direct Fit')
        else:
          space2 = np.append(np.linspace(1/(n), 1, 20)[:-1], 0.99999999)
          plt.plot(space, guesses, label = 'Direct Fit (Optimal)')
          plt.plot(space2, guesses2, label = 'Direct Fit (Nash)')

        if maximize == 1:
          title = f"β vs Socially Optimal Mixing Probability: n = {n} players"
        elif maximize == 2:
          title = f"β vs Optimal and Nash Mixing Probability: n = {n} players"
        else:
          title = f"β vs Mixed Nash Equilibrium Probability: n = {n+1} players"

        if maximize == 0:
          label = 'Newton\'s Method Appx. (Nash)'
        else:
          label = 'Newton\'s Method Appx. (Optimal)'
        plt.plot(space, guesses, "*", color = "green", label = label)
        if maximize == 2:
          space2 = np.append(np.linspace(1/(n), 1, 20)[:-1], 0.99999999)
          plt.plot(space2, guesses2, "*", color = "purple", label = 'Newton\'s Method Appx. (Nash)')

        fig.suptitle(title)
        ax.legend()
        plt.xlabel("β")
        plt.ylabel("π")
        ax.set_xlim([[1/(n+1),0][maximize > 0],1])
        ax.set_ylim([0.5,1])
        filex = BytesIO()
        fig.savefig(filex, format = "png")
        filex.seek(0)
        plt.close()
        images.append(filex)
        await msg.edit(content = f"{n - min(maximize, 1)} of 19 complete")

    imgs = [Image.open(image) for image in images]
    masterfilex = BytesIO()
    imgs[0].save(fp=masterfilex, format='GIF', append_images=imgs[1:], save_all=True, duration=200, loop=0)
    masterfilex.seek(0)
    await ctx.send(file=discord.File(masterfilex, "nash.gif"))

  @bot.command()
  async def anarchy(self, ctx):
    async with ctx.typing():
      payoffs = [[["1/2", "1/2"], ["1", "β"]], [["β", "1"], ["β/2", "β/2"]]]
      await ctx.send(gameformat(payoffs, nash = False, params = ["W1", "W2", "1", "2"]))

      pi, beta = symbols("pi beta")
      pinash = solve(pi/2+(1-pi) - (pi*beta+(1-pi)*beta/2), pi)
      await tex(ctx, str(latex(pinash)), "π value for mixed strategy Nash equilibrium:")

    async with ctx.typing():
      images = []
      for beta in np.linspace(0, 1, 20):
        expected = pi**2+2*(1-pi)*pi*(1+beta)+(1-pi)**2*beta
        mapper = lambdify(pi, expected, modules=['numpy'])
        xs = np.linspace(0, 1, 100)
        ys = mapper(xs)
        plt.clf()
        fig, ax = plt.subplots()
        plt.plot(xs, ys)

        fig.suptitle(f"Mixing Probability vs Expected Packets for β = {round(beta, 2)}")
        plt.xlabel("π")
        plt.ylabel("Packets")
        ax.set_xlim([0,1])
        ax.set_ylim([0,1.7])

        soln = solve(expected.diff(pi), pi) 
        value = expected.subs(pi, soln[0])
        plt.plot([soln[0]], [value], '.', color = "black")
        ax.text(soln[0], value, f"Maximum: ({round(float(soln[0]), 2)},{round(float(value), 2)})", horizontalalignment=['left', 'right'][float(soln[0]) > 0.5])

        pinashval = pinash[0].subs(symbols("beta"), beta)
        nashpackets = expected.subs(pi, pinashval)
        plt.plot([pinashval], [nashpackets], '.', color = "black")
        ax.text(pinashval, nashpackets, f"Nash Eq: ({round(float(pinashval), 2)},{round(float(nashpackets), 2)})", horizontalalignment=['left', 'right'][float(pinashval) > 0.5], verticalalignment="top")
        filex = BytesIO()
        fig.savefig(filex, format = "png")
        plt.close()
        filex.seek(0)
        images.append(filex)

      imgs = [Image.open(image) for image in images]
      masterfilex = BytesIO()
      imgs[0].save(fp=masterfilex, format='GIF', append_images=imgs[1:], save_all=True, duration=600, loop=0)
      masterfilex.seek(0)
      await ctx.send(file=discord.File(masterfilex, "expectation.gif"))


# from cs213bot, from cs221bot, from :b:ot
def _urlencode(*args, **kwargs):
  kwargs.update(quote_via=urllib.parse.quote)
  return urllib.parse.urlencode(*args, **kwargs)

requests.models.urlencode = _urlencode

async def tex(ctx, formula, msg = None):
  # from cs213bot, from cs221bot, from :b:ot
  formula = formula.strip("`")
  body = {
      "formula" : formula,
      "fsize"   : r"30px",
      "fcolor"  : r"FFFFFF",
      "mode"    : r"0",
      "out"     : r"1",
      "remhost" : r"quicklatex.com",
      "preamble": r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}",
      "rnd"     : str(random.random() * 100)
  }

  try:
    img = requests.post("https://www.quicklatex.com/latex3.f", data=body, timeout=10)
  except (requests.ConnectionError, requests.HTTPError, requests.TooManyRedirects, requests.Timeout):
    return await ctx.send("Render timed out.")

  if img.status_code != 200:
    return await ctx.send("Something went wrong. Maybe check your syntax?")

  if img.text.startswith("0"):
    await ctx.send(msg, file=discord.File(BytesIO(requests.get(img.text.split()[1]).content), "latex.png"))
  else:
    await ctx.send(" ".join(img.text.split()[5:]), delete_after=5)

def setup(bot):
  bot.add_cog(Anarchy(bot))