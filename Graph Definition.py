# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:09:23 2019

@author: ekgia
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


#creating first graphs
DG1 = nx.DiGraph()
DG2 = nx.DiGraph()

DG1.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
DG2.add_weighted_edges_from([(3, 1, 0.75), (1, 2, 0.5)])


#plotting graphs
plt.subplot(121)
nx.draw(DG1, with_labels=True, font_weight='bold')

plt.subplot(122)
nx.draw(DG2, with_labels=True, font_weight='bold')

plt.show()
