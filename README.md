# User Life Cycle Analysis in the _Overwatch_ Gaming Forum


Project for Berkeley NLP Course

Author: David Skarbrevik

You can read the corresponding [white paper](/final_paper.pdf) for this project.

Or alternatively explore the analysis in this [main jupyter notebook](/Overwatch_Forum_Analysis.ipynb).

### Main features of this repo currently:
* **final_paper.pdf** is the white paper of this project.

* **WebScrapping.ipynb** has a python class I made to gather data from Blizzard's forum sites. Blizzard uses the same html formatting for all their different forum sites so coding the scraping for one forum is pretty much coding for all their forums.

* **Overwatch_Forum_Analysis.ipynb** indepth processing and EDA of Overwatch forum posts.
* **utils.py** lots of great helper functions to process text, build language models and score those models.

***

### To do list (for myself):

* 1st priorities (must have):
  * add error bars to plots
  * isolate users that were initially active but then abandoned the forum
  * add some kind of backoff smoothing to LMs
  * consider different ways of filtering text for LMs (or not filtering at all)
  
* 2nd priorities (should have):
  * come up with ideas for baseline classifier (look at Danescu et al. for inspiration)
  * build simple baseline with sklearn
  
* 3rd priorities (nice to have):
  * scrape replies to posts (should drastically increase dataset, but links are not as predictable; need to figure that out)
  * develop automated framework for pulling new data from forum and updating "at risk" churn group from classifier
        * dashboard to display this information
  * integrate other forums (or at least other Blizzard forums... should try WoW next probably)
  * consider neural network based classifier
