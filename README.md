# Aoe2-NEAT-And-vgg19
A NEAT algorithm running on predictions from the VGG19 model to predict winning or losing state in the game age of empires 2 DE... The NEAT algorithm performs hotkeys and mouse actions based on the prediction from VGG19, with the aim of improving the fitness. Which improves if the score is higher than your enemy...

Chatgpt helped me write this..
it actually wrote most of the code while I checked it for errors.. and yea there were alot of errors so I feel like I did most of the job .. writing the right prompt isnt easy either!

All you need to do is run screenshotplaying.py and play the game.. choose one player color and stick with it... create two folders "Winning" and one called "Losing" and fill it up with scoreboard pictures of you losing and winning..
Then train the vgg19 model on the folder which has the two folders; "losing" and "winning".

Then you can train the neat algorithm.. I already provided a few hotkeys and mouse clicks and movements, as output.


****** Added neat_config file

I have a neat config file that is specifically made for this project.. it has the right amount of outputs and inputs, and all the other regulations is just standard as provided by chatgpt.

