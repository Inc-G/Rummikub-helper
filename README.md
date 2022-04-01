# Rummikub-helper
Helper for a variation of Rummikub, using tf object detection API

The input is a picture of the tiles on the table and a string of the tiles in your hand (hand = the blue rummikub holder for tiles). The output is True if you can play, together with what you can play and how, False otherwise.

The variation only involves how the jokers can be used. The valid sets are the same as those in classical Rummikub and the initial meld needs to have a value of 30 or more. But after you play the initial meld, there are only 3 rules.

(i) By the end of your turn, each tile on the table has to belong to a unique admissible set.
(ii) Each turn you must draw a single tile if and only if you don’t play, and
(iii) You cannot put a tile from the table to your hand.

This is slightly looser than the classical Rummikub, because you don’t have to specify which tile is the joker if it appears in a set with tiles of the same colour. For example, if on the table there are 3b, 4b, j and you have in your hand 1b, in classical Rummikub you cannot play if j represents a 5b. In this version you can as 1b, j, 3b, 4b is a valid set.

**USAGE**: Run Rummikub_main.py and follow the instructions. You need tensorflow >= 2.8, numpy, pandas, matplotlib. The modules and models folder should be in the same folder as Rummikub_main.py.

There are roughly two parts in this project: (1) an object detection and object classification part to get information from the pictures of the tiles on the table, and (2) a solver. Part (1) uses tensorflow object detection API together with two neural networks, and part (2) is a hard-coded python script. 

Part (1) consists of three neural networks. The first performs object detection using the Tensorflow object detection API: it locates Rummikub tiles in a photo. For each tile detected, you cut the photo along the tile detected, and pass it to two neural networks, one to detect the number (using a fine-tuned mobilenet + MLP prediction head, probably an overkill) and one to detect the colour (small MLP).

I trained the object detection neural network on 60 photos of tables as the ones in the folder ‘sample photos’. I trained the other two neural networks on less than 1k photos of tiles (using data augmentation), which are obtained by cutting a photo of a table along the tiles detected by the object detection neural network. I used the two notebooks in training_notebooks to classify a tile, and the object detection API to detect tiles.

Part (2) first detects all the valid sets that one can form using both the tiles on the table and those in your hand. Then it performs a variation of Knuth's Algorithm X to determine if you can play.

TO DO:
- The object detection part was trained on photos of tiles on a table, and struggles with photos of tiles in your hand. All photos were taken with a pixel phone. It probably makes sense to retrain the object detection neural network with a more diverse dataset (that includes photos of tiles in your hand).
- It would be cool to have a webapp to run all of this not from the terminal.
