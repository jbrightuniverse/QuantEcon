import discord
from discord.ext import commands as bot

import os

class Editor(bot.Cog):

    def __init__(self, bot):
        self.bot = bot
        self.channels = []

    @bot.command()
    async def commit(self, ctx):
        message = ctx.message.content[8:]
        if not message: return await ctx.send("Message?")

        text = os.popen(f"git add-commit -m '{message}'").read()
        if not text: text = "Failed."
        await ctx.send(text)

    @bot.command()
    async def push(self, ctx, branch):
        text = os.popen(f"git push origin {branch}").read()
        await ctx.send(f"Pushed.\nhttps://github.com/jbrightuniverse/QuantEcon")

    @bot.command()
    async def create(self, ctx, file):
        if file + ".py" in os.listdir("cogs"):
            return await ctx.send("Module already exists.")

        open("cogs/" + file + ".py", "a").close()
        await ctx.send(f"Created {file} module.")

    @bot.command()
    async def help(self, ctx):
        text = "QuantEcon is a bot designed for running various experiments and algorithms related to economics.\n\n"
        text += "Automatic help system coming soon.\n\n"
        text += "**API for Discord Python IDE**:\n\n"
        text += "`-reboot <module>`:\nreboots the module named `<module>`\n\n"
        text += "`-create <file>`:\ncreates a module named `<file`\n\n"
        text += "`-commit <message>`:\ncommit to origin with `<message>` commit message\n\n"
        text += "`-push <branch>`:\npush to <branch> branch\n\n"
        text += "`-edit`:\nlaunch the editor. Default file is \"dp.py\" for now. \n\nThe following commands require `-edit` to have already been called:\n"
        text += "`edit <number>`:\nset the cursor to line `<number>`.\n\n"
        text += "`add <number>\n<text>`:\nadd `<text>` with `<number>` indents to the current cursor pos.\n\n"
        text += "`tab <number>`:\nset indentation level of the current line to `<number>` indents.\n\n"
        text += "`delete <start> <end>`:\ndelete inclusively all lines from `<start>` to `<end>`\n\n"
        text += "`view`:\nview current code at cursor pos.\n"
        await ctx.send(text)

    @bot.command()
    async def edit(self, ctx, module = "dp"):
        if ctx.channel.id in self.channels: return await ctx.send("Session is already ongoing.")

        if ctx.guild.id != 761047255645945897 and ctx.author.id != 375445489627299851: return await ctx.send("Cannot use this command in this server!")

        if module + ".py" not in os.listdir("cogs"): return await ctx.send("Module not found.")

        self.channels.append(ctx.channel.id)

        lines = None
        with open(f"cogs/{module}.py") as f:
            lines = f.readlines()

        await display(ctx, lines, 0)

        editpointer = 1

        while True:
            message = await get(self.bot, ctx, "exit")
            if not message: 
                self.channels.remove(ctx.channel.id)
                return
            if message.content == "": continue

            if message.content.startswith("view"):
                if message.content.endswith("blank"):
                    await display(ctx, lines, editpointer-1, False)
                else:
                    await display(ctx, lines, editpointer - 1)

            elif message.content.startswith("edit"):
                try: 
                    line = message.content.split()[1]
                    if int(line) < 1 or int(line) > len(lines): raise Exception()
                    editpointer = int(line)
                    await display(ctx, lines, editpointer - 1)
                except: 
                    await ctx.send("Where's the line number?")
                    continue

            elif message.content.startswith("buffer"):
                amount = message.content.split()
                try:
                    if len(amount) == 2:
                        amount = int(amount[1])
                    else:
                        amount = 1

                    for i in range(amount):
                        lines.insert(editpointer - 1, "\n")

                    with open(f"cogs/{module}.py", "w") as f:
                        f.write(''.join(lines))
                    
                    await display(ctx, lines, editpointer - 1)

                except:
                    await ctx.send("Invalid amount.")
                    continue

            elif message.content.startswith("add"):
                text = message.content[3:]
                spaces = ""
                if message.content.split("\n")[0] != "add":
                    spaces = '  '*int(text.split("\n")[0])
                spaced = [spaces + a for a in text.split("\n")[1:]]
                for line in spaced:
                    lines.insert(editpointer - 1, line + "\n")
                    editpointer += 1

                with open(f"cogs/{module}.py", "w") as f:
                    f.write(''.join(lines))

                await display(ctx, lines, editpointer - 1)

            elif message.content.startswith("tab"):
                try: 
                    linedata = message.content.split()
                    if len(linedata) == 4:
                        start = int(linedata[2]) - 1
                        end = int(linedata[3])
                    else:
                        start = editpointer-1
                        end = editpointer
                    
                    line = linedata[1]

                    if int(line) < 0: raise Exception()

                    line = int(line)

                    for i in range(start, end):
                        lines[i] = '  '*line + lines[i].lstrip(' ')

                    with open(f"cogs/{module}.py", "w") as f:
                        f.write(''.join(lines))

                    await display(ctx, lines, editpointer - 1)

                except: 
                    await ctx.send("How many tabs?")
                    continue

            elif message.content.startswith("delete"):
                try:
                    line = message.content.split()
                    start = int(line[1])
                    if len(line) == 3:
                        end = int(line[2]) + 1
                    else:
                        end = start + 1

                    for i in range(start, end):
                        del lines[start - 1]

                    editpointer = start

                    with open(f"cogs/{module}.py", "w") as f:
                        f.write(''.join(lines))

                    await display(ctx, lines, editpointer - 1)

                except:
                    await ctx.send("One of those is not an int.")
                    continue


async def display(ctx, lines, editpointer, show_numbers = True):
    printlines = []
    start = max(0, editpointer - 15)
    if start != 0: 
        printlines.append("------------\n")

    end = min(editpointer + 15, len(lines))
    for i in range(start, end):
        if not show_numbers:
            printlines.append(lines[i])
        else:
            printlines.append(str(i+1).rjust(len(str(len(lines)))) + f"{[' ', '*'][i == editpointer]} " + lines[i])

    if end != len(lines): 
        printlines.append("------------")

    await ctx.send("``"+"`python\n" + ''.join(printlines) + "``"+"`")


async def get(bot, ctx, exitkey):
  try:
    message = await bot.wait_for("message", timeout = 600, check = lambda m: m.channel == ctx.channel)
  except:
    await ctx.send("Timed out waiting for you. Exiting.")
    return None

  if message.content == exitkey:
    await ctx.send("Exiting.")
    return None

  return message

    
def setup(bot):
    bot.add_cog(Editor(bot))