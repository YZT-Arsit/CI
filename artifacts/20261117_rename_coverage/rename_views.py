"""Change only entity-name assignments after the first matched epoch."""
import random


def views_for(batch, step):
    epoch = step // 18
    return [{**view, 'names': (view['names'] if epoch == 0 else
             random.Random(73117 + epoch * 1000 + batch['step'] * 2 + i).randrange(10**9))}
            for i, view in enumerate(batch['views'])]
