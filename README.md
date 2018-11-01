A Pokemon Showdown AI that uses Monte Carlo tree search with regret matching or Exp3 to make moves. It is assumed that both players have access to both team sheets. It supports most doubles and singles formats, including VGC. In VGC (and in all other formats), it assumes both players know which pokemon the opponent has in the back.

Requires the modified PS server from [https://github.com/samhippie/Pokemon-Showdown](https://github.com/samhippie/Pokemon-Showdown). Make sure you run `npm install` in the PS server directory before running this program.

The main MCTS algorithms used come from [http://mlanctot.info/files/papers/cig14-smmctsggp.pdf](http://mlanctot.info/files/papers/cig14-smmctsggp.pdf). The Regret Matching algorithm also uses some features from the DCFR algorithm in [https://arxiv.org/pdf/1809.04040.pdf](https://arxiv.org/pdf/1809.04040.pdf).
