A Pokemon Showdown AI that uses Monte Carlo tree search with regret matching or Exp3 to make moves. It is assumed that both players have access to both team sheets. It supports most doubles and singles formats, including VGC. In VGC (and in all other formats), it assumes both players know which pokemon the opponent has in the back.

Requires the modified PS server from [https://github.com/samhippie/Pokemon-Showdown](https://github.com/samhippie/Pokemon-Showdown). Make sure you run `npm install` in the PS server directory before running this program.

The main MCTS algorithms used come from [http://mlanctot.info/files/papers/cig14-smmctsggp.pdf](http://mlanctot.info/files/papers/cig14-smmctsggp.pdf). The Regret Matching algorithms also uses some features from the DCFR algorithm in [https://arxiv.org/pdf/1809.04040.pdf](https://arxiv.org/pdf/1809.04040.pdf). The CFR algorithm is based on the algorithm described in
[https://papers.nips.cc/paper/4569-efficient-monte-carlo-counterfactual-regret-minimization-in-games-with-many-player-actions.pdf](https://papers.nips.cc/paper/4569-efficient-monte-carlo-counterfactual-regret-minimization-in-games-with-many-player-actions.pdf), including the Average Sampling option. The use of on-policy rollouts for paths not taken in CFR is based on the use of probes in
[https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewFile/4937/5469](https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewFile/4937/5469).

Depends on pytorch, sqlitedict, psycopg (some of these can probably be removed)
Also depends on [https://github.com/YannDubs/Hash-Embeddings](https://github.com/YannDubs/Hash-Embeddings), and assumes that the hashembed directory can be found in $PYTHONPATH (i.e. import hashembed should work)
