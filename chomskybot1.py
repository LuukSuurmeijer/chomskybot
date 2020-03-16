import discord
import datetime
import nltk
import numpy
import re
from langdetect import detect_langs, detect, DetectorFactory
from corpus import Corpus
from ngram import BasicNgram
from discord.ext import commands
from discord.utils import get

with open('token', 'r') as f:
	TOKEN = f.read()

bot = commands.Bot(command_prefix='!')

#Datastructures for the language model of the chomskycorpus
c = Corpus("chomskycorpus.txt")
ngram = BasicNgram(4, c.tokens)



@bot.event
async def on_ready():
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('------')

@bot.event
async def on_disconnect():
	print(f'{datetime.datetime.now()} | Chomskybot disconnected')

@bot.event
async def on_connect():
	print(f'{datetime.datetime.now()} | Chomskybot reconnected')

@bot.event
async def on_message(message):
	channel = bot.get_channel(689082124577669135)
	if message.channel == channel:
		if message.author == bot.user or len(message.content.split(' ')) < 3 or message.content.startswith('!'):
			return

		langs = detect_langs(message.content)
		if 'nl' in [language.lang for language in langs]:
			return
		else:
			await message.delete()
			await channel.send(f'\"{message.content}\"\n{message.author.mention} praat Nederlands aub.')

	#need this otherwise it overrides commands
	await bot.process_commands(message)

@bot.command(brief='Returns pong')
async def ping(ctx):
    await ctx.send('Pong!')

@bot.command(brief='Returns current date and time')
async def time(ctx):
	now = datetime.datetime.now()
	await ctx.send(now.strftime("%d-%m-%Y %H:%M:%S"))

@bot.command(brief='Generate sentence using some Chomsky books as data.', description='Generate a random sentence using a 4gram of Chomsky\'s Understanding power and Language and Mind. Takes an integer as argument for the length of the sentence.')
async def quote(ctx, w: int, s=''):
	s = s.lower()
	try:
		s = c.sentgen(w, ngram, s)
		await ctx.send(f"\"{s.capitalize()}.\" - Chomskybot")
	except IndexError:
		await ctx.send("I've never said that before.")
	

bot.run(TOKEN)
