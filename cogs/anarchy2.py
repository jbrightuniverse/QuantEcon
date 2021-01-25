import discord
from discord.ext import commands as bot

import asyncio
import colorsys
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
import random
import requests
import time
import urllib.parse

from io import BytesIO
from PIL import Image, ImageDraw
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

class Anarchy2(bot.Cog):
  """:cloud_lightning: Hyperdimensional Price of Anarchy"""

  def __init__(self, bot):
    self.bot = bot

  @bot.command()
  async def generate_game(self, ctx):
    """
    temp
    """

    inputs = [int(a) for a in ctx.message.content[15:].split()]
    players = inputs.pop(0)
    routers = inputs.pop(0)
    
    def q_ary_n_seq(q, n):
      if n <= 1: 
        return [[k] for k in range(q)]

      sequences_to_add = q_ary_n_seq(q, n-1)
      cur_found = []
      for i in range(q):
        for j in sequences_to_add:
          cur_found.append([i] + j) 

      return cur_found

    betas = symbols(" ".join([f'beta{i}' for i in range(1, routers + 1)]))
    pis = symbols(" ".join([f'pi{i}' for i in range(1, routers + 1)]))
    all_routers = q_ary_n_seq(routers, players)
    equations = [0] * routers
    for outcome in all_routers:
      router = outcome[0]
      people_sharing = outcome.count(router)
      coefficient = betas[router]/people_sharing
      equations[router] += coefficient * np.prod([pis[i] for i in outcome[1:]])

    equationset = [equations[0] - i for i in equations[1:]] + [sum(pis) - 1]
    await tex(ctx, str(latex(equationset)))


  @bot.command()
  async def graph_poa(self, ctx, players, routers, social = ""):
    """
    temp
    """

    routers = int(routers)

    width = 800
    height = 600 
    seg = routers//2 + 1
    offset = (width/(seg+1))//2
    points = []

    for i in range(routers + 1):
      wd = width//(seg+1) * ((i//2) +1)
      hd = 3*height//4
      if i % 2 == 1:
        wd += int(round(offset))
        hd = height//4

      points.append((wd, hd))

    base = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(base)
    players = int(players)

    def q_ary_n_seq(q, n):
      if n <= 1: 
        return [[k] for k in range(q)]

      sequences_to_add = q_ary_n_seq(q, n-1)
      cur_found = []
      for i in range(q):
        for j in sequences_to_add:
          cur_found.append([i] + j) 

      return cur_found

    betas = symbols(" ".join([f'beta{i}' for i in range(1, routers + 1)]))
    pis = symbols(" ".join([f'pi{i}' for i in range(1, routers + 1)]))
    all_routers = q_ary_n_seq(routers, players)
    equations = [0] * routers
    total_payoff = 0
    for outcome in all_routers:
      router = outcome[0]
      people_sharing = outcome.count(router)
      coefficient = betas[router]/people_sharing
      equations[router] += coefficient * np.prod([pis[i] for i in outcome[1:]])
      for i in range(len(outcome)):
        total_payoff += betas[outcome[i]] * np.prod([pis[j] for j in outcome])/outcome.count(outcome[i])

    equationset = [equations[0] - i for i in equations[1:]] + [sum(pis) - 1]
    diffmatrix = np.array([
      [diff(eq, pi) for pi in pis] for eq in equationset
    ])

    fail = False
    async def shaderecursive(level, prev, probabilities):
      nonlocal fail
      wd = points[level+1][0]
      hd = points[level+1][1]
      sdensity = 2
      try: density = max(3, int(round(routers * 30/((routers-1)**sdensity))))
      except:
        return

      for k in range(density + 1):
        await asyncio.sleep(0)
        prob = k/density
        newprobabilities = probabilities[:]
        newprobabilities.append(prob)

        cx = wd - (1-prob)*(wd-prev[0])
        cy = hd - (1-prob)*(hd-prev[1])
        mid = [int(round(cx)), int(round(cy))]

        if level == routers - 1:
          if not social:
            initial_pis = [1/routers for i in range(routers)]
            a = time.time()
            try:
              while True:
                if time.time() - a > 0.25:
                  fail = True
                  raise Exception()
                cursub = list(zip(pis, initial_pis)) + list(zip(betas, newprobabilities))
                subber = np.vectorize(lambda x: x.subs(cursub))
                A = subber(diffmatrix)
                b = -1 * np.array([eq.subs(cursub) for eq in equationset], dtype='float')
                delta = np.linalg.solve(A.astype(np.float64), b)
                initial_pis += delta
                if (np.abs(delta) <= 0.001 * np.ones(routers)).all(): break

              expected_packets = simplify(total_payoff.subs(list(zip(pis, initial_pis)) + list(zip(betas, newprobabilities))))
              
            except:
              expected_packets = 0
          else:
            def f(mypis):
              nonlocal pis, total_payoff, betas, newprobabilities
              return -1 * total_payoff.subs(list(zip(pis, mypis)) + list(zip(betas, newprobabilities)))

            res = optimize.minimize(f, [1/routers for i in range(routers)], constraints=[LinearConstraint([np.ones(routers)], [1], [1])], bounds = Bounds(np.zeros(routers), np.inf * np.ones(routers)))
            expected_packets = simplify(total_payoff.subs(list(zip(pis, res.x)) + list(zip(betas, newprobabilities))))

          col = colorsys.hls_to_rgb(expected_packets/players, 0.5, 1)
          col = [int(round(255*col[0])), int(round(255*col[1])), int(round(255*col[2])), 255]
          r, g, b, _ = base.getpixel((mid[0], mid[1])) 
          if (r, g, b) != (255, 255, 255):
            col[0] = (col[0] + r)//2
            col[1] = (col[1] + g)//2
            col[2] = (col[2] + b)//2
          constant = 10
          draw.ellipse([(mid[0]-constant, mid[1]-constant), (mid[0]+constant, mid[1]+constant)], fill =tuple(col), outline =tuple(col)) 
          #base.putpixel(mid, tuple(col))

        else:
          await shaderecursive(level+1, mid, newprobabilities)

    await shaderecursive(0, points[0], [])
    if fail:
      await ctx.send("ERROR: One of the computations did not finish in time. Results may be inaccurate.")
    filex = BytesIO()
    base.save(filex, 'PNG')
    filex.seek(0)
    await ctx.send(file = discord.File(filex, 'betas.png'))


  @bot.command()
  async def poa(self, ctx):
    """
    temp
    """

    # take numerical inputs, set an initial guess of evenly distributed pi, attempt to iterate Newton's method until convergence of result

    betas = [float(a) for a in ctx.message.content[5:].split()]
    if any(b <= 0 or b > 1 for b in betas[1:]):
      return await ctx.send("Invalid router probability detected. Range is (0, 1].")

    players = int(betas.pop(0))
    routers = len(betas)
    
    def q_ary_n_seq(q, n):
      if n <= 1: 
        return [[k] for k in range(q)]

      sequences_to_add = q_ary_n_seq(q, n-1)
      cur_found = []
      for i in range(q):
        for j in sequences_to_add:
          cur_found.append([i] + j) 

      return cur_found

    #betas = symbols(" ".join([f'beta{i}' for i in range(1, routers + 1)]))
    pis = symbols(" ".join([f'pi{i}' for i in range(1, routers + 1)]))
    all_routers = q_ary_n_seq(routers, players)
    equations = [0] * routers
    total_payoff = 0
    for outcome in all_routers:
      router = outcome[0]
      people_sharing = outcome.count(router)
      coefficient = betas[router]/people_sharing
      equations[router] += coefficient * np.prod([pis[i] for i in outcome[1:]])
      for i in range(len(outcome)):
        total_payoff += betas[outcome[i]] * np.prod([pis[j] for j in outcome])/outcome.count(outcome[i])

    def f(mypis):
      nonlocal pis, total_payoff
      return -1 * total_payoff.subs(list(zip(pis, mypis)))
    res = optimize.minimize(f, [1/routers for i in range(routers)], constraints=[LinearConstraint([np.ones(routers)], [1], [1])], bounds = Bounds(np.zeros(routers), np.inf * np.ones(routers)))
    await ctx.send(res.x)
    await ctx.send(sum(res.x))

    equationset = [equations[0] - i for i in equations[1:]] + [sum(pis) - 1]
    diffmatrix = np.array([
      [diff(eq, pi) for pi in pis] for eq in equationset
    ])

    initial_pis = [1/routers for i in range(routers)]
    a = time.time()
    while True:
      await asyncio.sleep(0)
      if time.time() - a > 10:
        return await ctx.send("10 seconds have passed; iteration did not converge in time.")
      cursub = list(zip(pis, initial_pis))
      subber = np.vectorize(lambda x: x.subs(cursub))
      A = subber(diffmatrix)
      b = -1 * np.array([eq.subs(cursub) for eq in equationset], dtype='float')
      delta = np.linalg.solve(A.astype(np.float64), b)
      initial_pis += delta
      if (np.abs(delta) <= 0.001 * np.ones(routers)).all(): break

    await ctx.send(initial_pis)
    await ctx.send("Sum = " + str(sum(initial_pis)))
    

    # http://homepage.math.uiowa.edu/~whan/3800.d/S7-3.pdf


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
  bot.add_cog(Anarchy2(bot))