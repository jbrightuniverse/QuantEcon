import discord
from discord.ext import commands as bot

import asyncio
import itertools
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
from sympy.matrices import Matrix
import sympy

import scipy.optimize as optimize
from scipy.optimize import LinearConstraint, Bounds

from util import *

import warnings
warnings.filterwarnings("ignore")

class Anarchy(bot.Cog):
  """:desktop: Price of Anarchy Experiment"""

  def __init__(self, bot):
    self.bot = bot


  @bot.command()
  async def routergraph(self, ctx):
    """
    Graphs expected packet functions for the Price of Anarchy with 2 players and m routers.

    **Usage:**
    `-routeroptimal [beta1] [beta2] [beta3] [...]`

    **Examples:**
    `-routeroptimal 0.5 0.5 0.5` Gives function for a 4 router setup with 3 β = 0.5 routers and one variable router
    """

    basebetas = [float(a) for a in ctx.message.content[13:].split()]
    betas = basebetas + [symbols("beta")]
    routers = len(betas)

    equation2 = (2*routers - 1) * np.prod(betas) / (sum(np.prod(i) for i in itertools.combinations(betas, routers - 1)))
    plt.clf()
    fig, ax = plt.subplots()

    mapper = lambdify(betas[-1], equation2, modules=['numpy'])
    xs = np.linspace(0, 1, 100)
    ys = mapper(xs)
    plt.plot(xs, ys, label="Nash Equilibrium")

    def f(pis):
      nonlocal newbetas, routers
      return -1 * (sum(newbetas[i]*pis[i]**2 for i in range(routers)) + 2 * (sum(pis[r[0]]*pis[r[1]]*(newbetas[r[0]]+newbetas[r[1]]) for r in itertools.combinations(range(routers), 2))))
    
    newbetas = []
    expected_packets = []
    for beta in np.linspace(0, 1, 100):
      newbetas = basebetas + [beta]
      res = optimize.minimize(f, [1/routers for i in range(routers)], constraints=[LinearConstraint([np.ones(routers)], [1], [1])], bounds = Bounds(np.zeros(routers), np.inf * np.ones(routers)))
      mpis = res.x
      expected_packets.append(sum(newbetas[i]*mpis[i]**2 for i in range(routers)) + 2 * (sum(mpis[r[0]]*mpis[r[1]]*(newbetas[r[0]]+newbetas[r[1]]) for r in itertools.combinations(range(routers), 2))))

    plt.plot(np.linspace(0, 1, 100), expected_packets, label="Socially Optimal")

    fig.suptitle(f"2 Players and {len(betas)} routers: β in {[round(beta, 2) for beta in betas[:-1]]} + variable")
    ax.legend()
    plt.xlabel("β (variable router)")
    plt.ylabel("Expected Packets")
    ax.set_xlim(0,1)
    ax.set_ylim(0,2)
    filex = BytesIO()
    fig.savefig(filex, format = "png")
    filex.seek(0)
    plt.close()
    await ctx.send(file=discord.File(filex, "manyrouters.png"))

  """
  @bot.command()
  async def routeroptimal(self, ctx, size):
    Computes the socially optimal expected packet functions for the Price of Anarchy with 2 players and m routers.

    **Usage:**
    `-routeroptimal <routers>`

    **Examples:**
    `-routeroptimal 2` Gives function for a 2 router setup

    routers = int(size)
    betas = symbols(" ".join([f'beta{i}' for i in range(1, routers + 1)]))
    sets = list(itertools.combinations(betas, routers - 1))
    denom = sum(np.prod(i) for i in sets.copy())
    numer = sum(sum(np.prod(i)*beta for beta in i) for i in sets.copy()) + (1 - (routers-2)*(routers-1)) * np.prod(betas)
    equation = numer/denom
    await tex(ctx, str(latex(equation)), "Result:")
  """

  @bot.command()
  async def routeroptimalpi(self, ctx, size):
    """
    Computes the socially optimal π probability values for the Price of Anarchy with 2 players and m routers.

    **Usage:**
    `-routeroptimalpi [beta1] [beta2] [...]`

    **Examples:**
    `-routeroptimalpi 0.5 0.5` Gives functions for a 2 router setup
    """

    betas = [float(a) for a in ctx.message.content[17:].split()]
    routers = len(betas)

    def f(pis):
      nonlocal betas, routers
      return -1 * (sum(betas[i]*pis[i]**2 for i in range(routers)) + 2 * (sum(pis[r[0]]*pis[r[1]]*(betas[r[0]]+betas[r[1]]) for r in itertools.combinations(range(routers), 2))))
    res = optimize.minimize(f, [1/routers for i in range(routers)], constraints=[LinearConstraint([np.ones(routers)], [1], [1])], bounds = Bounds(np.zeros(routers), np.inf * np.ones(routers)))
    await ctx.send(res.x)
    await ctx.send(sum(res.x))
    

  @bot.command()
  async def multirouter(self, ctx, size):
    """
    Computes the Nash equilibrium expected packet functions for the Price of Anarchy with 2 players and m routers.

    **Usage:**
    `-multirouter <routers>`

    **Examples:**
    `-multirouter 2` Gives function for a 2 router setup
    """

    routers = int(size)
    betas = symbols(" ".join([f'beta{i}' for i in range(1, routers + 1)]))
    equation = (2*routers - 1) * np.prod(betas) / (sum(np.prod(i) for i in itertools.combinations(betas, routers - 1)))
    await tex(ctx, str(latex(equation)), "Result:")


  @bot.command()
  async def multirouterpacket(self, ctx, size):
    """
    Computes the generic expected packet functions for the Price of Anarchy with 2 players and m routers.

    **Usage:**
    `-multirouterpacket <routers>`

    **Examples:**
    `-multirouterpacket 2` Gives function for a 2 router setup
    """

    routers = int(size)
    if routers > 6:
      return await ctx.send("Too big.")

    betas = symbols(" ".join([f'beta{i}' for i in range(1, routers + 1)]))
    pis = symbols(" ".join([f'pi{i}' for i in range(1, routers + 1)]))
    packetcounter = sum(simplify(betas[i] * pis[i]**2) for i in range(routers)) + sum(simplify(sum(simplify(2*pis[i]*pis[j]*(betas[i]+betas[j])) for j in range(i))) for i in range(1, routers))
    await tex(ctx, str(latex(simplify(packetcounter))), "Result:")

  @bot.command()
  async def multirouternash(self, ctx, size):
    """
    Computes the Nash equilibrium π probability values for the Price of Anarchy with 2 players and m routers.

    **Usage:**
    `-multirouternash <routers>`

    **Examples:**
    `-multirouternash 2` Gives π values for a 2 router setup
    """

    routers = int(size)
    if routers > 6:
      return await ctx.send("Too big.")

    betas = symbols(" ".join([f'beta{i}' for i in range(1, routers + 1)]))

    thelist = []
    for i in range(routers - 1):
      newlist = [-betas[0]/2] + [0 for j in range(1, routers)]
      newlist[i+1] = betas[i+1]/2
      thelist.append(newlist)

    thelist.append([1 for i in range(routers)])
    A = Matrix(thelist)
    thelist2 = []
    for i in range(1, routers):
      thelist2.append(betas[i] - betas[0])

    thelist2.append(1)
    b = Matrix(routers, 1, thelist2)
    x = A.LUsolve(b)
    text = "```\n"
    text += "\n\n".join([str(simplify(x[i])) for i in range(routers)])
    await ctx.send(text[:1993] + "...```")

  @bot.command()
  async def beta(self, ctx):
    """
    Computes the disparity between Nash equilibrium and socially optimial expected packet counts for the Price of Anarchy with 2 routers and n computers for n between 2 and 20.

    Takes the results and compiles them into a GIF.

    This command uses Newton's method of polynomial root approximation to determine the solutions to the n-1th degree polynomial system of equations dictating the expected payoffs for each player in a mixed strategy.

    **Usage:**
    `-beta`
    """

    msg = await ctx.send("Working...")
    async with ctx.typing():
      images = []
      ns = []
      for n in range(2, 21):
        pi, beta = symbols("pi beta")
        equation = sum((-1)**(k+1) * pi**k * binomial(n-1, k) for k in range(0, n-1)) + (beta + (-1)**n)*pi**(n-1)
        eqdiff = diff(equation, pi)
        packetcounter = beta + 1 - beta*pi**n - (1-pi)**n

        guesses = []
        for beta in np.append(np.linspace(1/(n+1), 1, 40)[:-1], 0.99999999):
          cur_guess = beta
          while True:
            await asyncio.sleep(0)
            new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
            
            if abs(new_cur_guess - cur_guess) <= 0.001:
              break

            cur_guess = new_cur_guess

          guesses.append(simplify(packetcounter.subs(pi, re(cur_guess)).subs(symbols("beta"), beta)))

        guesses2 = []
        nx = n - 1
        equation = sum((symbols("beta") - (-1)**k * binomial(nx+1, k+1)) * pi**k for k in range(0, nx+1))
        eqdiff = diff(equation, pi)
        for beta in np.append(np.linspace(1/(n+1), 1, 40)[:-1], 0.99999999):
          cur_guess = beta
          while True:
            await asyncio.sleep(0)
            new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
            
            if abs(new_cur_guess - cur_guess) <= 0.001:
              break

            cur_guess = new_cur_guess

          guesses2.append(simplify(packetcounter.subs(pi, re(cur_guess)).subs(symbols("beta"), beta)))

        disparities = [g1-g2 for g1, g2 in zip(guesses, guesses2)]
        ns.append(disparities)
        await msg.edit(content = f"{n - 1} of 19 complete: part 1 of 2")

      i = 0
      for beta in np.append(np.linspace(1/(n+1), 1, 40)[:-1], 0.99999999):
        plt.clf()
        fig, ax = plt.subplots()
        plt.plot(range(2, 21), [disp[i] for disp in ns], label = 'Disparity')
        plt.fill_between(np.array(range(2, 21), dtype=float), np.array([disp[i] for disp in ns], dtype=float))
        fig.suptitle(f"Number of Players vs Disparity Between Optimal and Nash Solutions: β = {round(beta, 2)}")
        ax.legend()
        plt.xlabel("Players")
        plt.ylabel("Disparity (Packets)")
        ax.set_xlim(2,20)
        ax.set_ylim(0,0.5)
        filex = BytesIO()
        fig.savefig(filex, format = "png")
        filex.seek(0)
        plt.close()
        images.append(filex)
        await msg.edit(content = f"{i + 1} of 19 complete: part 2 of 2")
        i += 1
        

    imgs = [Image.open(image) for image in images]
    masterfilex = BytesIO()
    imgs[0].save(fp=masterfilex, format='GIF', append_images=imgs[1:], save_all=True, duration=200, loop=0)
    masterfilex.seek(0)
    await ctx.send(file=discord.File(masterfilex, "nash.gif"))


  @bot.command()
  async def price(self, ctx):
    """
    Computes the disparity between Nash equilibrium and socially optimial expected packet counts for the Price of Anarchy with 2 routers and n computers for n between 2 and 20.

    Takes the results and compiles them into a GIF.

    This command uses Newton's method of polynomial root approximation to determine the solutions to the n-1th degree polynomial system of equations dictating the expected payoffs for each player in a mixed strategy.

    **Usage:**
    `-price`
    """

    msg = await ctx.send("Working...")
    async with ctx.typing():
      images = []
      for n in range(2, 21):
        pi, beta = symbols("pi beta")
        equation = sum((-1)**(k+1) * pi**k * binomial(n-1, k) for k in range(0, n-1)) + (beta + (-1)**n)*pi**(n-1)
        eqdiff = diff(equation, pi)
        packetcounter = beta + 1 - beta*pi**n - (1-pi)**n

        guesses = []
        for beta in np.append(np.linspace(1/(n+1), 1, 40)[:-1], 0.99999999):
          cur_guess = beta
          while True:
            await asyncio.sleep(0)
            new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
            
            if abs(new_cur_guess - cur_guess) <= 0.001:
              break

            cur_guess = new_cur_guess

          guesses.append(simplify(packetcounter.subs(pi, re(cur_guess)).subs(symbols("beta"), beta)))

        guesses2 = []
        nx = n - 1
        equation = sum((symbols("beta") - (-1)**k * binomial(nx+1, k+1)) * pi**k for k in range(0, nx+1))
        eqdiff = diff(equation, pi)
        for beta in np.append(np.linspace(1/(n+1), 1, 40)[:-1], 0.99999999):
          cur_guess = beta
          while True:
            await asyncio.sleep(0)
            new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
            
            if abs(new_cur_guess - cur_guess) <= 0.001:
              break

            cur_guess = new_cur_guess

          guesses2.append(simplify(packetcounter.subs(pi, re(cur_guess)).subs(symbols("beta"), beta)))

        disparities = [g1-g2 for g1, g2 in zip(guesses, guesses2)]

        plt.clf()
        fig, ax = plt.subplots()
        space = np.append(np.linspace(1/(n+1), 1, 40)[:-1], 0.99999999)
        plt.plot(space, disparities, label = 'Disparity')
        plt.fill_between(np.array(space, dtype=float), np.array(disparities, dtype=float))
        fig.suptitle(f"β vs Disparity Between Optimal and Nash Solutions: n = {n} players")
        ax.legend()
        plt.xlabel("β")
        plt.ylabel("Disparity (Packets)")
        ax.set_xlim(0,1)
        ax.set_ylim(0,0.5)
        filex = BytesIO()
        fig.savefig(filex, format = "png")
        filex.seek(0)
        plt.close()
        images.append(filex)
        await msg.edit(content = f"{n - 1} of 19 complete")

    imgs = [Image.open(image) for image in images]
    masterfilex = BytesIO()
    imgs[0].save(fp=masterfilex, format='GIF', append_images=imgs[1:], save_all=True, duration=200, loop=0)
    masterfilex.seek(0)
    await ctx.send(file=discord.File(masterfilex, "nash.gif"))


  @bot.command()
  async def polynomial(self, ctx):
    """
    Computes the π probability values for the Price of Anarchy with 2 routers and n computers.
    
    This command uses Newton's method of polynomial root approximation to determine the solutions to the n-1th degree polynomial system of equations dictating the expected payoffs for each player in a mixed strategy.

    **Usage:**
    `-polynomial <num_players> [max | all] [packet]`

    **Examples:**
    `-polynomial 2`     Displays β vs π for Nash equilibrium π given 2 players
    `-polynomial 3 max` Displays β vs π for socially optimal π given 3 players
    `-polynomial 5 all` Gives Nash eq. and socially optimal π given 5 players

    Add `packet` at the end of the command to display the graphs in terms of expected packets instead of π. 
    """

    if ctx.message.content.endswith(" packet"):
      ctx.message.content = ctx.message.content[:-7]
      packet = True
    else:
      packet = False

    if ctx.message.content.endswith("max"):
      ctx.message.content = ctx.message.content[:-3]
      maximize = True
    elif ctx.message.content.endswith("all"):
      ctx.message.content = ctx.message.content[:-3]
      maximize = 2
    else:
      maximize = False

    async with ctx.typing():
      try:  
        n = int(ctx.message.content[12:])
      except:
        return await ctx.send("not valid")
      if n > 50: return await ctx.send("too big")
      if n < 2: return await ctx.send("too small")

      if not maximize: n -= 1

      pi, beta = symbols("pi beta")
      if maximize:
        equation = sum((-1)**(k+1) * pi**k * binomial(n-1, k) for k in range(0, n-1)) + (beta + (-1)**n)*pi**(n-1)
      else:
        equation = sum((beta - (-1)**k * binomial(n+1, k+1)) * pi**k for k in range(0, n+1))

      eqdiff = diff(equation, pi)
      packetcounter = beta + 1 - beta*pi**n - (1-pi)**n
      packetcounter2 = beta + 1 - beta*pi**(n+1) - (1-pi)**(n+1)

      guesses = []
      for beta in np.append(np.linspace([1/(n+1),0][maximize > 0], 1, [20, 40][maximize > 0])[:-1], 0.99999999):
        cur_guess = beta
        while True:
          await asyncio.sleep(0)
          new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
          
          if abs(new_cur_guess - cur_guess) <= 0.001:
            break

          cur_guess = new_cur_guess

        if packet:
          if maximize == 0: pc = packetcounter2
          else: pc = packetcounter
          guesses.append(simplify(pc.subs(pi, re(cur_guess)).subs(symbols("beta"), beta)))
        else:
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

          if packet:
            guesses2.append(simplify(packetcounter.subs(pi, re(cur_guess)).subs(symbols("beta"), beta)))
          else:
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

        if maximize == 0: pc = packetcounter2
        else: pc = packetcounter
        if packet: exp = simplify(pc.subs(pi, exp))
        mapper = lambdify(beta, exp, modules=['numpy'])
        xs = np.linspace(0, 1, 100)
        ys = mapper(xs)
        plt.plot(xs, ys, label=label)

        if maximize == 2:
          exp = (2-beta)/(1+beta)
          if packet: exp = simplify(packetcounter.subs(pi, exp))
          mapper = lambdify(beta, exp, modules=['numpy'])
          xs = np.linspace(0, 1, 100)
          ys = mapper(xs)
          plt.plot(xs, ys, label='Symbolic Answer (Nash)')

      elif n == 2 + min(maximize, 1):
        beta = symbols("beta")
        if maximize:
          exp = (sympy.sqrt(beta) - 1)/(beta - 1)
          if packet: exp = simplify(packetcounter.subs(pi, exp))
          label = 'Symbolic Answer (Optimal)'
        else:
          exp = ((3+beta) - sympy.sqrt(-3*beta**2 + 22*beta - 3)) / (2*(1-beta)) 
          if packet: exp = exp = simplify(packetcounter2.subs(pi, exp))
          label = 'Symbolic Answer (Nash)'

        mapper = lambdify(beta, exp, modules=['numpy'])
        xs = np.linspace(0, 1, 100)
        ys = mapper(xs)
        plt.plot(xs, ys, label=label)

        if maximize == 2:
          exp = ((3+beta) - sympy.sqrt(-3*beta**2 + 22*beta - 3)) / (2*(1-beta))
          if packet: exp = simplify(packetcounter.subs(pi, exp))
          mapper = lambdify(beta, exp , modules=['numpy'])
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
        title = f"β vs Socially Optimal {['Mixing Probability', 'Packet Count'][packet]}: n = {n} players"
      elif maximize == 2:
        title = f"β vs Optimal and Nash {['Mixing Probability', 'Packet Count'][packet]}: n = {n} players"
      else:
        title = f"β vs Mixed Nash Equilibrium {['Probability', 'Packet Count'][packet]}: n = {n+1} players"

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
      plt.ylabel(["π", "Expected Packets"][packet])
      ax.set_xlim([[1/(n+1),0][maximize > 0 or packet],1])
      ax.set_ylim([[0.5, 0][packet],[1, 2][packet]])
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
    `-polyall [max | all] [packet]`

    **Examples:**
    `-polyall`     Displays β vs π for Nash equilibrium π
    `-polyall max` Displays β vs π for socially optimal π
    `-polyall all` Gives Nash eq. and socially optimal π

    Add `packet` at the end of the command to display the graphs in terms of expected packets instead of π. 
    """

    if ctx.message.content.endswith(" packet"):
      ctx.message.content = ctx.message.content[:-7]
      packet = True
    else:
      packet = False

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
        packetcounter = beta + 1 - beta*pi**n - (1-pi)**n
        packetcounter2 = beta + 1 - beta*pi**(n+1) - (1-pi)**(n+1)

        guesses = []
        for beta in np.append(np.linspace([1/(n+1),0][maximize > 0], 1, [20, 40][maximize > 0])[:-1], 0.99999999):
          cur_guess = beta
          while True:
            await asyncio.sleep(0)
            new_cur_guess = simplify(cur_guess - (simplify(equation.subs(pi, cur_guess).subs(symbols("beta"), beta)) / simplify(eqdiff.subs(pi, cur_guess).subs(symbols("beta"), beta))))
            
            if abs(new_cur_guess - cur_guess) <= 0.001:
              break

            cur_guess = new_cur_guess

          if packet:
            if maximize == 0: pc = packetcounter2
            else: pc = packetcounter
            guesses.append(simplify(pc.subs(pi, re(cur_guess)).subs(symbols("beta"), beta)))
          else:
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

            if packet:
              guesses2.append(simplify(packetcounter.subs(pi, re(cur_guess)).subs(symbols("beta"), beta)))
            else:
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
          title = f"β vs Socially Optimal {['Mixing Probability', 'Packet Count'][packet]}: n = {n} players"
        elif maximize == 2:
          title = f"β vs Optimal and Nash {['Mixing Probability', 'Packet Count'][packet]}: n = {n} players"
        else:
          title = f"β vs Mixed Nash Equilibrium {['Probability', 'Packet Count'][packet]}: n = {n+1} players"

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
        plt.ylabel(["π", "Expected Packets"][packet])
        ax.set_xlim([[1/(n+1),0][maximize > 0 or packet],1])
        ax.set_ylim([[0.5, 0][packet],[1, 2][packet]])
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
    """
    Price of Anarchy basic problem.

    **Usage:**
    `-anarchy`
    """

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