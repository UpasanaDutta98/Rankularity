#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import multiprocessing
import time
import pickle
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from random import sample
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
#import profile
#import pstats

# In[2]:




def random_ranking(n):
    ranking = {}
    chooseList = [i for i in range(0,n)]
    for i in range(len(chooseList)):
        x = random.choice(chooseList)
        ranking[i] = x
        chooseList.remove(x)
    return ranking


# In[3]:


def violations(ranking, edgeList): ##Given a ranking, find the number of violations.
    # edgeList is a dictionary of tuple of edge -> its weight
    number_of_violations = 0
    for (i,j) in edgeList:
        if ranking[i] < ranking[j]:
            number_of_violations+=edgeList[(i,j)]
    return number_of_violations


# In[4]:


def get_weighted_edgeDict(AdjList):
    edgeWeight_dict = {}
    weighted_edgeList = []
    for i in range(len(AdjList)):
        for edge_and_weight in AdjList[i]:
            weighted_edgeList.append((i,edge_and_weight[0],edge_and_weight[1]))
    for i in range(len(weighted_edgeList)):
        edgeWeight_dict[(weighted_edgeList[i][0],weighted_edgeList[i][1])] = weighted_edgeList[i][2]

    return edgeWeight_dict


# In[5]:


def find_communityOrder(collapsed_AdjList): # Here G is a collapsed graph version of the original graph, and is weighted and directed.
    n = len(collapsed_AdjList)
    ranking = random_ranking(n)
    #print('initial ranking = ', ranking)
    
    edgeWeight_dict = get_weighted_edgeDict(collapsed_AdjList)
    
    current_number_of_violations = violations(ranking, edgeWeight_dict)
    #print("initial violations = ", current_number_of_violations)

    number_of_passes = 0
    t = 0
    time = []
    violation = []
    number_of_violations = -1
    #print("max number of passes = ", n*(n-1)/2)
    while number_of_passes < (50 * n): # Here, we keep the max passes as 5n and not nC2 because the number of 
        # communities in the collapsed graph would be pretty low, and so 5n is taken so as to ensure that
        # sufficient number of checks for swaps have been done before returning the the final order.
        
        time.append(t)
        t+=1
        violation.append(current_number_of_violations)
        
        #Choose 2 random nodes and propose their swap.
        nodeList = [i for i in range(n)]
        x = random.choice(nodeList)
        nodeList.remove(x)
        y = random.choice(nodeList)
        
        #print("nodes chosen for swapping = ", x, y)
        rank_X = ranking[x]
        rank_Y = ranking[y]
        temp = ranking[x]
        ranking[x] = ranking[y]
        ranking[y] = temp
        #print('swapped ranking = ', ranking)
        number_of_violations = violations(ranking, edgeWeight_dict)
        #print('violations after swap = ', number_of_violations)
        if number_of_violations > current_number_of_violations:
            ## we do not change the ordering, so we revert back.
            #print('revert back')
            ranking[x] = rank_X
            ranking[y] = rank_Y
            number_of_passes+=1
        elif number_of_violations == current_number_of_violations:
            ## we keep the swapped ordering (swap is accpeted)
            #print('is same')
            number_of_passes+=1
        else:
            ## we keep the swapped ordering
            #print('keep the swapped order')
            number_of_passes = 0
            current_number_of_violations = number_of_violations
            #print("number of passes set to 0")
    #print("Final violations = ", current_number_of_violations)    
    #print('number of passes = ',number_of_passes)
    #plt.plot(time, violation) #, label = label_name)#, linestyle='dashed')
    #plt.xlabel('Time elapsed')
    #plt.ylabel('Number of violations')
    #print("ranking before reversing = ", ranking)
    #print("Then we change every value to n - value.")
    reversed_ranking = {}
    for key, value in ranking.items():
        reversed_ranking[key] = n - value - 1
    #print("final ranking = ", reversed_ranking)
    return reversed_ranking


# In[6]:


def collapse_graph(AdjList, communities, c): #G is a directed, unweighted graph
    
    number_of_communities = c
    collapsed_AdjList = [[] for k in range(number_of_communities)]
    
    weight_between_communities = [[0 for k in range(number_of_communities)] for k in range(number_of_communities)]
   
    for i in range(len(AdjList)):
        for eachnode in AdjList[i]:
            weight_between_communities[communities[i]][communities[eachnode]] += 1
            
    
    for i in range(number_of_communities):
        for j in range(number_of_communities):
            #if weight_between_communities[i][j] != 0:
            collapsed_AdjList[i].append((j,weight_between_communities[i][j]))
     
    #print("collapsed Adjlist = ", collapsed_AdjList)
    return collapsed_AdjList


# In[7]:


# You have to remove all networkx, so store everything in your own data structure. Even for nodes you do not call the
# G.nodes() function, rather you save them at the beginning and never compute this n more than once. Also you don't 
# use G.add_edges() because that line is executed a lot of times. You just do not use the 'G.' anywhere.


# In[8]:


def B_val(G, u, v, rho, flag_model):
    w = -1
    if G.has_edge(u, v):
        w = 1
    else:
        w = 0
    m = G.number_of_edges()
    
    value = -1
    
    if flag_model == 0:
        value = (w - ((G.out_degree[u] * G.in_degree[v])/(m)))/(m)
    else:
        value = (w - rho)/m
    
    #print(u, v, value)
    return value


# In[9]:


# @do_cprofile
def get_B_matrix(AdjList, rho, flag_model, degree_in, degree_out):
    B_matrix = []
    number_of_nodes = len(AdjList)
    number_of_edges = 0
    for node in range(number_of_nodes):
        number_of_edges += degree_out[node]
            
    if flag_model == 1: ## use Erdos-Renyi Null model
        B_matrix = [[-rho for i in range(number_of_nodes)] for j in range(number_of_nodes)]
        for i in range(number_of_nodes):
            for eachnode in AdjList[i]:
                B_matrix[i][eachnode] += 1
            
    elif flag_model == 0: ## use Configuration Null model
        #print("before adding 1", B_matrix)
        for i in range(number_of_nodes):
            B_list = []
            for j in range(number_of_nodes):
                #print('for node ', i, j, 'we have B value = ', -((degree_out[i]*degree_in[j])/number_of_edges))
                B_list.append(-((degree_out[i]*degree_in[j])/number_of_edges))
            B_matrix.append(B_list)
                
        for i in range(number_of_nodes):
            for eachnode in AdjList[i]:
                B_matrix[i][eachnode] += 1
        #print("after adding 1", B_matrix) 
                
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            B_matrix[i][j] = B_matrix[i][j]/number_of_edges
    #print("before return ", B_matrix)
    return B_matrix


# In[10]:


## b_{rs} = sum_over_{ij} [B_{ij}.delta_{gi,r}.delta_{gj,s}]
## where B_{ij} = 1/m * [A_{ij} - (k-out_{i} * k-in_{j})/m]

## this function performs similar to nx.networkx.algorithms.community.modularity for modularity

def brs_matrix(AdjList, community, rho, flag_model, c, degree_in, degree_out):
    # *** this function works only for unweighted and directed graphs***
    
    # input : AdjList of the graph
    #       : community is a n-sized list of the community that each of the n nodes belong to, 
    #       : community index must start from 0 **
    
    # output : returns the b_rs matrix using the above formula
    #t11 = time.time()
    #print("brs matrix start time = ", t11)
    number_of_communities = c
    number_of_nodes = len(community)
    
    b_rs = [[0 for k in range(number_of_communities)] for i in range(number_of_communities)]
    
    # Now, we make a list A of lists, where list A is of size c and each of the lists store the nodes that belong to 
    # each of the c communities. Hence the r_list and the s_list will not be re-computed in every iteration like we 
    # are doing now.
    
    partitions = [[] for i in range(c)]
    for k in range(len(community)):
        partitions[community[k]].append(k)

    # Get the B_matrix computed here, this function should be called only once in an entire run of the full program.
    B_matrix = get_B_matrix(AdjList, rho, flag_model, degree_in, degree_out)
    #print(B_matrix)
        
    for r in range(number_of_communities):
        for s in range(number_of_communities):
            r_list = partitions[r]
            s_list = partitions[s]
            sum_B = 0
            for i, j in product(r_list, s_list):
                    sum_B += B_matrix[i][j]    
            b_rs[r][s] = sum_B
    #t22 = time.time()
    #print("brs matrix end time = ", t22)
    #print('total time to build the brs matrix for', G.number_of_nodes(), 'nodes = ', (t22-t11)/60, 'mins.')
    return b_rs, B_matrix


# In[11]:


def calculate_U(b_rs, comm_count):
    # output : returns the summation of b_rs values across groups i.e for all b_rs with r<s
    
    number_of_communities = comm_count
    
    sum_u = 0
    for r in range(number_of_communities-1):
        for s in range(r+1,number_of_communities):
            sum_u += b_rs[r][s]
            
    return sum_u        


# In[12]:


def calculate_D(b_rs, comm_count):
    # output : returns the summation of b_rs values within groups i.e for all b_rs with r=s.

    number_of_communities = comm_count
    sum_d = 0
    for r in range(number_of_communities):
            sum_d += b_rs[r][r]
            
    return sum_d   


# In[13]:


def change_in_U(number_of_nodes, number_of_communities, i, old_comm, new_comm, community, partition_dict, B_matrix):
    '''
    U is basically the sum of the upper triangle of the b_rs matrix.
    
    if new_comm(i) > old_comm(i), then.........
    there is no affect in overall U for the edges entering i from the nodes belonging to community less than 
    old_comm(i), because they were already counted before and they should be counted now too. For edges coming from 
    community >= old_comm(i) and less than new_comm(i) to i, all the *B values* will now get added to the overall U,
    because these were not counted in overall U previously. There is no affect for edges coming from community
    >= new_comm(i) to node i in the overall U value, because they were anyway not counted previously and should not 
    be counted now too.
    The edges going from i to all the community <=old_comm(i) were not counted in overall U previously and should 
    not be counted now too. The edges going from i to all the community > old_comm(i) and <= new_comm(i) were
    counted previously, but should not be counted now. So all those *B values* should now be subtracted from overall
    U. The edges going from i to all the community greater than new_comm(i) should not have any affect in overall U
    because they were already counted before and they should be counted now too.
    
    
    if new_comm(i) < old_comm(i), then.........
    there is no affect in overall U for the edges entering i from the nodes belonging to community less than 
    new_comm(i), because they were already counted before and they should be counted now too. For edges coming from 
    community >= new_comm(i) and less than old_comm(i) to i, all the *B values* now needs to be subtracted from the
    overall U, because these were counted in overall U previously, but should not be counted now. There is no affect 
    for edges coming from community >= old_comm(i) to node i in the overall U value, because they were anyway 
    not counted previously and should not be counted now too.
    The edges going from i to all the community <=new_comm(i) were not counted in overall U previously and should 
    not be counted now too. The edges going from i to all the community > new_comm(i) and <= old_comm(i) were not
    counted previously, but should added to overall U now. So all those *B values* should be added to the overall
    U. The edges going from i to all the community greater than old_comm(i) should not have any affect in overall U
    because they were already counted before and they should be counted now too.
    
    Here we just record the change, i.e the delta value.
    '''
    delta = 0
    delta_test = 0
    
    if new_comm > old_comm:
        for k in range(number_of_communities):
            if k >= old_comm and k < new_comm:
                for eachnode in partition_dict[k]:
                    #print("eachnode = ", eachnode)
                    if eachnode!=i:
                        delta_test = delta_test + B_matrix[eachnode][i]

#         for node in range(number_of_nodes):
#             if node != i:
#                 #print('neighbour = ',node)
#                 if community[node] >= old_comm and community[node] < new_comm:
#                     #print("node = ", node)
#                     #print('we will add the B value for this neighbour')
#                     delta = delta + B_matrix[node][i]
        

        
        for k in range(number_of_communities):
            if k > old_comm and k <= new_comm:
                for eachnode in partition_dict[k]:
                    #print("eachnode = ", eachnode)
                    delta_test = delta_test - B_matrix[i][eachnode]
                    
#         for node in range(number_of_nodes):
#             if node!=i:
#                 #print('neighbour = ',node)
#                 if community[node] > old_comm and community[node] <= new_comm:
#                     #print('we will subtract the B value for this neighbour')
#                     #print("node = ", node)
#                     delta = delta - B_matrix[i][node]
        
                
    elif new_comm < old_comm:
        #print('we have new_comm < old_comm')
        #print('checking each node from the list - ', list(G.nodes()))
        for k in range(number_of_communities):
            if k >= new_comm and k < old_comm:
                for eachnode in partition_dict[k]:
                    #print("eachnode = ", eachnode)
                    delta_test = delta_test - B_matrix[eachnode][i]

#         for node in range(number_of_nodes):
#             if node != i:
#                 #print('neighbour = ',node)
#                 if community[node] >= new_comm and community[node] < old_comm:
#                     #print('we will subtract the B value for this neighbour')
#                     #print("node = ", node)
#                     delta = delta - B_matrix[node][i]

        for k in range(number_of_communities):
            if k > new_comm and k <= old_comm:
                for eachnode in partition_dict[k]:
                    #print("eachnode = ", eachnode)
                    if eachnode!=i:
                        delta_test = delta_test + B_matrix[i][eachnode]
            
            
#        for node in range(number_of_nodes):
#             if node != i:
#                 #print('neighbour = ',node)
#                 if community[node] > new_comm and community[node] <= old_comm:
#                     #print('we will add the B value for this neighbour')
#                     #print("node = ", node)
#                     delta = delta + B_matrix[i][node]
#        print("\n")
    #print("change in U ",delta, delta_test)            
    return delta_test











#     if new_comm > old_comm:
#         #print('we have new_comm > old_comm')
#         #print('checking each node from the list - ', list(G.nodes()))
#         for node in range(number_of_nodes):
#             if node != i:
#                 #print('neighbour = ',node)
#                 if community[node] >= old_comm and community[node] < new_comm:
#                     #print('we will add the B value for this neighbour')
#                     delta = delta + B_matrix[node][i]
#         #print('checking each node from the list - ', list(G.nodes())) 
#         for node in range(number_of_nodes):
#             if node!=i:
#                 #print('neighbour = ',node)
#                 if community[node] > old_comm and community[node] <= new_comm:
#                     #print('we will subtract the B value for this neighbour')
#                     delta = delta - B_matrix[i][node]
                
#     elif new_comm < old_comm:
#         #print('we have new_comm < old_comm')
#         #print('checking each node from the list - ', list(G.nodes()))
#         for node in range(number_of_nodes):
#             if node != i:
#                 #print('neighbour = ',node)
#                 if community[node] >= new_comm and community[node] < old_comm:
#                     #print('we will subtract the B value for this neighbour')
#                     delta = delta - B_matrix[node][i]
                
#         #print('checking each node from the list - ', list(G.nodes()))    
#         for node in range(number_of_nodes):
#             if node != i:
#                 #print('neighbour = ',node)
#                 if community[node] > new_comm and community[node] <= old_comm:
#                     #print('we will add the B value for this neighbour')
#                     delta = delta + B_matrix[i][node]
#     #print(delta)            
#     return delta


# In[14]:


def change_in_D(number_of_nodes, number_of_communities, i, old_comm, new_comm, community, partition_dict, B_matrix):
    '''
    From the overall D, we can subtract the values which results from i leaving its old community and add the
    values which result from i joining its new community.
    
    We subtract all the *B values* for the edges going from i to all the nodes of community old_comm(i) and the edges
    entering i from the nodes of community old_comm(i), and we add all the *B values* for the edges going from i to 
    all the nodes of community new_comm(i) and the edges entering i from the nodes of community new_comm(i).
    '''
    delta = 0
    delta_test = 0
    #print("here i = ", i)
    for k in range(number_of_communities):
        if k == old_comm:
            for eachnode in partition_dict[k]:
                #print("old eachnode = ", eachnode)
                delta_test = delta_test - B_matrix[eachnode][i]
                delta_test = delta_test - B_matrix[i][eachnode]
        elif k == new_comm:
            for eachnode in partition_dict[k]:
                #print("new eachnode = ", eachnode)
                delta_test = delta_test + B_matrix[eachnode][i]
                delta_test = delta_test + B_matrix[i][eachnode]
                
    #print("\n\n\n")            
#     for node in range(number_of_nodes):
#         #if node!=i:
#         if community[node] == old_comm:
#             #print("old node = ", node)
#             delta = delta - B_matrix[node][i]
#             delta = delta - B_matrix[i][node]
#         elif community[node] == new_comm:
#             #print("new node = ", node)
#             delta = delta + B_matrix[node][i]
#             delta = delta + B_matrix[i][node]
            
#     for node in range(number_of_nodes):
#         #if node!=i:
#         #print("node = ", node)
#         if community[node] == old_comm:
#             print("old node = ", node)
#             delta = delta - B_matrix[i][node]
#         elif community[node] == new_comm:
#             print("new node = ", node)
#             delta = delta + B_matrix[i][node]
    #print("\n\n\n") 
    #print("Change in D ",delta, delta_test)
    #print("\n\n")
    return delta_test


# In[15]:


def change_in_R(number_of_nodes, number_of_communities, i, old_comm, new_comm, community, alpha, partition_dict, B_matrix):
    delta_U = change_in_U(number_of_nodes, number_of_communities, i, old_comm, new_comm, community, partition_dict, B_matrix)
    
    delta_D = change_in_D(number_of_nodes, number_of_communities, i, old_comm, new_comm, community, partition_dict, B_matrix)
    
    delta_R = (alpha*delta_U) + ((1 - alpha)*delta_D)
    
    #print('delta_R = ', delta_R)
    return delta_R


# In[16]:


def calculate_rankularity(community, b_rs, alpha):
    # output : returns the R value of a given partition
    number_of_communities = len(set(community))
    U = calculate_U(b_rs, number_of_communities)
    D = calculate_D(b_rs, number_of_communities)
    
    R = (alpha*U) + ((1 - alpha)*D)
    
    return R


# ### Modifying Aaron's code for locally greedy heuristic

# In[17]:


def drawGz(G,z):
    # DO NOT MODIFY THIS FUNCTION
    # This function draws G with node labels from partition z
    #
    # input  : G is a networkx graph
    #        : z is n sized list of integers on interval [1,c], a partition of G's nodes
    # output : none
    # 
    # WARNING: function is optimistic: assumes inputs are properly formatted
    #print(z.shape,"in draw")
    colors = ['#d61111','#11c6d6','#d67711','#11d646','#1b11d6','#d611cc'] # map node labels to colors (for the visualization)

    node_colors = []
    for i in G.nodes():
        #print(f'z[{i}] = {int(z[i])}')
        node_colors.append(colors[int(z[i])])
    nsize  = 600
    flabel = True

    if G.order() > 50:
        nsize  = 100
        flabel = False
        
    nx.draw_networkx(G,with_labels=flabel,node_size=nsize,width=2,node_color=node_colors,pos=nx.circular_layout(G)) # draw it pretty
    limits=plt.axis('off')                                      # turn off axes
    plt.show() 

    return


# In[18]:


def makeAMove(number_of_nodes, community, fr, alpha, c, old_R, partition_dictionary, B_matrix):
    # For each non 'frozen' node in the current partition, this function tries all (c-1) possible group moves for it
    # It returns the combination of [node i and new group r] that produces the best log-likelihood over the non-frozen set.
    
    # input  : AdjList of the input graph
    #        : community is a list showing the partition of G's nodes with a community label at each index.
    #        : fr is an n-sized binary list of frozen nodes, where 0 = not frozen and 1 = frozen.
    #        : alpha is a constant used in the Rankularity calculation
    #        : rho is used if the Null is the ER Model
    #        : flag_model = 0 for CM and 1 for ER
    
    # output : bestR, the best Rankularity found
    #        : bestMove, [i,r] the node i and new group r to achieve bestL
    
    #b_rs = brs_matrix(G, community, rho, flag_model)
    differences = []
    best_R    = -np.inf
    # ***** bestR SHOULD be made to -infinity because makeAMove HAS to for sure suggest for 
    # the best move out of all the (n-t)(c-1) possible single-node moves, for t nodes frozen (moved) so far. 
    
    bestMove = [-1, -1] # [node i, group r] assignment for bestR

    # c is the number of communities.
    
    for i in range(number_of_nodes):
        if fr[i] == 0:   # if i is not a 'frozen' node
            s = community[i]    #  the current community label of i
            for r in range(c): #  try all the groups
                
                if r != s:     #    except the current one
                    #community[i] = r   #    move i to group r
                    
                    current_R = old_R + change_in_R(number_of_nodes, c, i, s, r, community, alpha, partition_dictionary, B_matrix)
                    #print("current_R in MAKE = ", current_R)
                    # calculate change in R for this new partition
                    current_R = round(current_R, 10)
                    best_R = round(best_R, 10)
                    if best_R!=-np.inf:
                        differences.append(current_R - best_R)

                    if best_R == -np.inf: # because -np.inf cannot be converted to integer.
                        best_R    = current_R  #  best Rankularity so far
                        bestMove = [i,r,s]  #  the move that gets us there
                        
                    elif current_R > best_R:     # 
                    #elif current_R - best_R > 0.0000001:
                        best_R    = current_R  #  best Rankularity so far
                        bestMove = [i,r,s]  #  the move that gets us there
                        #print('best_R in MAKE is updated to ', best_R)
            #community[i]= s   # put i back where we found it
                ##### do not modify below here #####  
                
    #print("Best to move node",bestMove[0],"from community", community[bestMove[0]], "to community",bestMove[1])
    #print('..which will make the R', bestR)
    return best_R,bestMove,differences


# In[19]:


## Problem is in makeaMove, at the end all becomes 0,
## so size of set of community list becomes 1, so best_R stays at minus infinity, minus infinity is returned, and 
## bestMove has -1 community in it, current Rankularity =  -inf untill the next round phase is started, but then
## again the same thing happens.


# In[20]:


def plotLL(LL,pc):
    # DO NOT MODIFY THIS FUNCTION
    # This function makes a nice plot of the log-likelihood trajectory
    #
    # input  : LL is list of log-likelihood values of length (n+1)*(pc+1)
    #        : pc is the phase counter
    # output : none
    # 
    # WARNING: function is optimistic: assumes inputs are properly formatted

    n   = int(len(LL)/(pc+1)-1) # calculate n from the size of LL
    tc  = len(LL)               # number of partitions considered in the LL trajectory

    fig = plt.figure()
    ax1 = fig.add_subplot(111) # put multiple 
    
    plt.plot(range(tc), LL, 'b.-', alpha=0.5)  # plot the log-likelihood trajectory
    for i in range(pc+1):                      # add vertical bars to demarcate phases, add black squares for phase-maxima
        plt.plot([(i)*(n+1),(i)*(n+1)],[min(LL),max(LL)], 'k--', alpha=0.5)
        LLp = LL[(i)*(n+1):(i+1)*(n+1)]
        b = LLp.index(max(LLp))
        plt.plot([(i)*(n+1)+b],[max(LLp)], 'ks', alpha=0.5)

    plt.ylabel('Rankularity score')
    plt.xlabel('number of steps considered')
    plt.show()


# In[21]:


def random_z(n,c):
    # input  : number of nodes n, and number of groups c
    # output : returns a random partition z (n x 1), in which z_i = Uniform(0,c-1)

    import random as rnd
    rnd.seed()
    
    z = [1 for i in range(n)]
    '''for i in range(10):
        z[i] = 0'''
        

    ##### do not modify above here #####
    ### YOUR CODE
    
    partition_dictionary = defaultdict(set)
    for i in range(n):
        community = rnd.randint(0,c-1)
        z[i] = int(community)
        partition_dictionary[int(community)].add(i)
        
    

    ##### do not modify below here #####

    return z,partition_dictionary


# In[22]:


def local_greedy_heuristic(AdjList, alpha_value, number_of_communities, rho, flag_model, degree_in, degree_out):
    T = 30
    n = len(AdjList)
    c = number_of_communities
    alpha = alpha_value
    makeAmove_00 = []
    makeAmove_06 = []
    localgreedy_00 = []
    localgreedy_06 = []
    
    
    community, partition_dictionary  = random_z(n,c) #[0,1,2,0,1,2,1,2,2,0,1]   # initial partition
    
    Rankularity_values = []                # Rankularity over the entire algorithm (via .append)

    current_partition = community        # partition to start with.
    #print("randomised starting partition = ", current_partition)
    b_rs, B_matrix = brs_matrix(AdjList, community, rho, flag_model, c, degree_in, degree_out)  # b_rs matrix to start with
    #print(b_rs)
    best_R = calculate_rankularity(current_partition, b_rs, alpha)  # initial Rankularity score
    #print("starting rankularity for randomised partition = ", best_R)
    #print('starting bestR = ', bestR)
    #Rankularity_values.append(best_R)            # track Rankularity

    # 3.0 the main loop setup
    #print(f'phase[0] initial z, R = {bestR}')
                
    #drawGz(G,current_partition)

    best_partition = community
    best_partition_dictionary = partition_dictionary

    t  = 1  # counter for number of partitions considered, in this phase
    pc = 0  # counter for number of phases completed

    while True:
        f = [0 for i in range(n)]  # no nodes frozen
        flag_updated = 0   # flag becomes 1 if we find at least one new move in the entire phase that increases our R.
        current_partition = copy.deepcopy(best_partition)
        current_partition_dictionary = copy.deepcopy(best_partition_dictionary)
        ############
        current_R = best_R
        t = 0
        ############
        for j in range(n):
            choice_R,choiceMove,differences = makeAMove(n, current_partition, f, alpha, c, current_R, current_partition_dictionary, B_matrix)
            #print("choice_R returned by makeAMove = ", choice_R)
            choice_R = round(choice_R,10)
            best_R = round(best_R,10)
            i = choiceMove[0]    # move node i
            r = choiceMove[1]    # to community r
            s = choiceMove[2]    # from community s
            f[i]    = 1          # freeze node i, so frozen list f is updated here
            current_partition[i] = r          # make the greedy choice
            current_partition_dictionary[s].remove(i)
            current_partition_dictionary[r].add(i)
            current_R = choice_R
            #print("current_changed partition = ", current_partition)

            # Remember, freezing a node and updating its community is mandatory in every iteration.
            Rankularity_values.append(current_R) # track Rankularity (even if it decreases)
            if alpha == 0:
                localgreedy_00.append(current_R - best_R)
            elif alpha == 0.6:
                localgreedy_06.append(current_R - best_R)
            #if int(choice_R*10000) - int(best_R*10000) > 1:  # track best R and partition in the phase
            if choice_R > best_R:  # track best R and partition in the phase
                #print(choice_R, best_R, choiceMove)
                best_R = choice_R  # update the best Rankularity we have yet (among all phases)
                best_partition = copy.deepcopy(current_partition)  #
                best_partition_dictionary = copy.deepcopy(current_partition_dictionary)
                #print("best_R in localGreedy updated to ", best_R, "with best partition as ", best_partition)
                #print("\n\n\n YAYY \n\n")
                flag_updated = 1
                #print('new choiceR = ', choiceR)
                #print("bestR is updated to", bestR)
                #print("best Community is updated to", best_partition)

            t  = t  + 1          # increment step counter
            #time22 = time.time()
            #print('t = ',t, "took ", (time22-time11)/60, "mins.")
            #print(t)
            #if t%5 == 0:
                #print("step counter inside a phase (out of 200 steps) = ", t)
            
            
        # End end of each phase, check convergence criteria; if not converged, setup for next phase
        if flag_updated==0:
            # HALT: no better partition was found for this phase
            #print(f' --> WE HAVE CONVERGENCE due to no updation <-- ')
            break
        elif pc>=T:
            # HALT: we've run out of phases
            #print(f' --> WE HAVE CONVERGENCE due to running out of phase<-- ')
            break
        else:
            pc = pc + 1          # increment phase counter
            #print("pc = ", pc)
            

            
    #print(f'phase[{pc}], final_R =', bestR)
    #drawGz(G,best_partition)
    #print("Final best partiton = ", best_partition)
    #plotLL(Rankularity_values,pc)
    #print("final pc = ", pc)
    #print(best_partition)
    #drawGz(G,best_partition)
    return best_partition, makeAmove_00, makeAmove_06, localgreedy_00, localgreedy_06



# In[23]:


def fraction_of_correct_nodes(ground_truth, actual):
    correct = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] == actual[i]: # if this node's order was correctly identified
            correct+=1
            
    fraction = correct/len(ground_truth)
    
    #nmi = sklearn.metrics.normalized_mutual_info_score(ground_truth, actual)

    '''if fraction > (1 - fraction):
        print("correct fraction = ", fraction)
        return fraction
    else:
        print("correct fraction = ",1 - fraction)
        return 1-fraction'''
    
    #print('fraction correct = ', fraction)
    return fraction


# In[42]:


def generate_graph(SBM_matrix, p,list_of_numberOfNodes):
    group_label = []
    for num in range(len(list_of_numberOfNodes)):
        for x in range(list_of_numberOfNodes[num]):
            group_label.append(num)

    total_number_of_nodes = sum(list_of_numberOfNodes)
    total_edges = 0
    AdjList = [set() for k in range(total_number_of_nodes)]
    degree_out = [0 for i in range(total_number_of_nodes)]
    degree_in = [0 for i in range(total_number_of_nodes)]

    ## Try adding an edge between each pair of nodes
    for n1 in range(total_number_of_nodes):
        for n2 in range(total_number_of_nodes):
            if n1!=n2:
                community1 = group_label[n1]
                community2 = group_label[n2]


                random_num = random.random()
                if random_num <= 1:         # with probability 1, edge is decided by SBM structure (no more structurelessness)
                    #print(SBM_matrix,community1,community2 )
                    egde_probability = SBM_matrix[community1][community2]
                    rand_for_edge = random.random()
                    if rand_for_edge <= egde_probability:
                        total_edges+=1
                        AdjList[n1].add(n2)
                        degree_out[n1]+=1
                        degree_in[n2]+=1

                else:                     # with probability 1-p, edge is added from structurelessness
                    egde_probability = rho
                    rand_for_edge = random.random()
                    if rand_for_edge <= egde_probability:

                        AdjList[n1].add(n2)
                        degree_out[n1]+=1
                        degree_in[n2]+=1


    rho = total_edges / (total_number_of_nodes**2 - total_number_of_nodes)
    return AdjList, rho, degree_in, degree_out




def run_Rankularity(unique_id,AdjList, rho, degree_in, degree_out, list_of_numberOfNodes, alpha, c, flag_model, SBM_matrix):
    initial_time = time.time()
    #alpha = alpha_value
    number_of_communities = c    
    differences_makeAmove_00 = []
    differences_makeAmove_06 = []
    differences_localgreedy_00 = []
    differences_localgreedy_06 = []
    
    ground_truth = []
    for num in range(len(list_of_numberOfNodes)):
        for x in range(list_of_numberOfNodes[num]):
            ground_truth.append(num)
            
    b_rs, B_matrix = brs_matrix(AdjList, ground_truth, rho, flag_model, c, degree_in, degree_out)  # b_rs matrix to start with
    best_R = calculate_rankularity(ground_truth, b_rs, alpha)  # initial Rankularity score
    ground_truth_Rankularity = best_R
            
    flag = 1
    if flag == 1:
        ground_truth_order_same = 1
    elif flag == 0:
        ground_truth_order_same = 0
        

    p = 1
    best_partition, makeAmove_00, makeAmove_06, localgreedy_00,localgreedy_06 = local_greedy_heuristic(AdjList, alpha, number_of_communities, rho, flag_model, degree_in, degree_out)
    differences_makeAmove_00 += makeAmove_00
    differences_makeAmove_06 += makeAmove_06
    differences_localgreedy_00 += localgreedy_00
    differences_localgreedy_06 += localgreedy_06

#     collapsed_Graph = collapse_graph(AdjList, best_partition, number_of_communities)
#     finalOrder = find_communityOrder(collapsed_Graph)
    
#     final_answer = []
#     for community in best_partition:
#         final_answer.append(finalOrder[community])
#     b_rs, B_matrix = brs_matrix(AdjList, final_answer, rho, flag_model, c, degree_in, degree_out)  # b_rs matrix to start with
#     best_R = calculate_rankularity(final_answer, b_rs, alpha)  # initial Rankularity score
#     final_answer_Rankularity = best_R

    fractionCorrect = fraction_of_correct_nodes(ground_truth, best_partition) # Think on this, is this correct?
    NMI = normalized_mutual_info_score(ground_truth, best_partition) 
    #print("alpha = ", alpha, "GT = ", ground_truth, "FA = ", final_answer)
    coef_sp, pvalue_sp = spearmanr(ground_truth, best_partition) 
    coef_kt, pvalue_kt = kendalltau(ground_truth, best_partition) 
    coef_ps, pvalue_ps = pearsonr(ground_truth, best_partition) 
    confusion = confusion_matrix(ground_truth, best_partition)
    node_count = defaultdict(int)
    for i in range(len(best_partition)):
        node_count[best_partition[i]] += 1
    #print("alpha = ", alpha, ", Community order = ", finalOrder, ", fraction correct = ", fractionCorrect) #, ", Node counts = ", node_count)
#     flag = 1
#     for eachcommunity in finalOrder:
#         if eachcommunity != finalOrder[eachcommunity]:
#             flag = 0
#             break
#     if flag == 1:
#         same_order_as_MVR = 1
#     elif flag == 0:
#         same_order_as_MVR = 0
    
    print("Completed alpha = ", alpha)  
    #print("alpha = ", alpha, "Spearman = ", coef_sp, pvalue_sp)
    return fractionCorrect,coef_kt,coef_sp,coef_ps,SBM_matrix,list_of_numberOfNodes,confusion, NMI,best_partition, coef_ps, pvalue_ps, coef_sp, pvalue_sp, coef_kt, pvalue_kt


from collections import defaultdict
import numpy as np


# In[26]:


def get_SBM_matrix(k, a, b):
############# Because the K math assumes that the same a, b, and c is used all over the matrix ################
    SBM_matrix = [[0 for com in range(k)] for com in range(k)]
    for i in range(k):
        # Upper triangular elements of the row
        for j in range(i+1, k):
            SBM_matrix[i][j] = b #round(np.random.normal(b, 0.1, None),3)

        # Diagonal element of the row
        SBM_matrix[i][i] = a # round(np.random.normal(a, 0.1, None),3)

        # Lower triangular elements of the row
        j = 0
        while j < i:
            c = 0.1 #np.random.normal(0.1, 0.05, None)
            if c >=0 and c < SBM_matrix[j][j] and c < SBM_matrix[j][i] and c < SBM_matrix[i][i]:
                SBM_matrix[i][j] = round(c,3)
            else:
                j-=1
            j+=1

    print("Built SBM matrix for k = ", k)
    return SBM_matrix

def get_performance_varyingAlpha(iteration, a, b, return_dict):
    Accuracies_06 = []
    Kendalltau_06 = []
    k = 6
    alpha = 0
    start = 1
    order_flag = 0
    while order_flag == 0:
        nodes = random.randint(20, 200)
        list_of_numberOfNodes = [nodes for comsize in range(k)] # For K math, all communities are of same size.
        SBM_matrix = get_SBM_matrix(k, a, b)
        AdjList, rho, degree_in, degree_out = generate_graph(SBM_matrix, 1, list_of_numberOfNodes) 

        # Check if the generated graph satisfies our definition of "order" or not (which is MVR based, but for communities)
        ground_truth = []
        number_of_communities = k
        for num in range(len(list_of_numberOfNodes)):
            for x in range(list_of_numberOfNodes[num]):
                ground_truth.append(num)

        collapsed_Graph = collapse_graph(AdjList, ground_truth, number_of_communities)
        finalOrder = find_communityOrder(collapsed_Graph)
        order_flag = 1
        for eachcommunity in finalOrder:
            if eachcommunity != finalOrder[eachcommunity]:
                order_flag = 0
                break
        if order_flag == 0:
            print("Order definition not satisfied")
            continue
            
    while alpha <= 1:   
        print("iteration = ", iteration, ", alpha = ", alpha, list_of_numberOfNodes)
        returned = run_Rankularity(start,AdjList, rho, degree_in, degree_out, list_of_numberOfNodes,alpha,k,1,SBM_matrix)
        Accuracies_06.append(returned[0])
        Kendalltau_06.append(returned[1])

        alpha = alpha + 0.05
        alpha = round(alpha, 2)
        
    return_dict[iteration] = [Accuracies_06, Kendalltau_06]
    

def get_RankularityPerformance(a, b):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    All_kendalls = []
    All_acc = []
    NMIs_06 = []
    Confusions_06 = []
    FinalOrder_06 = []
    FinalOrder_00 = []
    RankularityScores_06 = []
    RankularityScores_00 = []
    NodeList_06 = []
    NodeList_00 = []
    SBM_information_06 = []
    SBM_information_00 = []
    iterations = 0
    while iterations < 50:
        p = multiprocessing.Process(target=get_performance_varyingAlpha, args=(iterations, a, b, return_dict))
        jobs.append(p)
        p.start()
        iterations += 1
        
    for proc in jobs:
        proc.join()
    return return_dict.values()

# Completed running (20, 200) for 2Eq cases so check them, currently running (20,100) for k=6 cases on Fiji.
# I think we should stick to (20,200) for 2 community cases and  k = 6 case for the presentation.
# Currently running k=6 case for (20, 200) on Fiji.
# Changing alpha position with (20, 100) brought no change, so currently running with (30,100) again, if still not good change to fixed values instead of normal because I know that gave clean plots. If that gives clean plots again, then it is because of detectability, so stick to fixed values when showing range of alpha and use normal distributions when only running for alpha = 0.6. When finalised, start runs for (20, 200) and leave on run. You would still have your plots on k=6 from your previous slides.

# Today night check size 20 for numcomm=10 case, if good, start the (20, 200) case on any k case on Fiji. If not, let the results be for (30, 100) and update smaller runs to 30 too.
# For 2 unequal community case, it is okay even we have to run simulations with (20, 500) because 2 communities do not take much time. If simulations not possible, use the landscape pictures from the draft.


def final_call(a, b):
    t1 = time.time()
    returned_values = get_RankularityPerformance(a, b)
    pickle_out = open("KmathVerification_K6_"+str(a)+"_"+str(b)+"_withCI.pkl","wb")
    pickle.dump(returned_values, pickle_out)
    pickle_out.close()    
    t2 = time.time()
    print('time taken = ', (t2-t1)/60, 'mins.')
    


import sys

a = float(sys.argv[1])
b = float(sys.argv[2])

final_call(a, b)