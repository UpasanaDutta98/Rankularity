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
    
    between_edges = 0  # Number of edges in between communities.
    
    for i in range(len(collapsed_AdjList)):
        for j in range(len(collapsed_AdjList)):
            if i != j:
                between_edges += collapsed_AdjList[i][j][1]
        
       
        
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
    while number_of_passes < 0: #(20 * n): # Here, we keep the max passes as 5n and not nC2 because the number of 
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

    if between_edges == 0:
        between_edges = 1
    fractional_final_violation = current_number_of_violations / between_edges
    if fractional_final_violation == 0:
        fractional_final_violation = 0.01
    directionality = between_edges/fractional_final_violation
    return reversed_ranking, between_edges, fractional_final_violation, directionality


# In[6]:


def collapse_graph(AdjList, communities, c): #G is a directed, unweighted graph
    
    number_of_communities = c
    collapsed_AdjList = [[] for k in range(number_of_communities)]
    
    weight_between_communities = [[0 for k in range(number_of_communities)] for k in range(number_of_communities)]
   
    for i in range(len(AdjList)):
        for eachnode in AdjList[i]:
            weight_between_communities[communities[i]][communities[eachnode[0]]] += eachnode[1]
            
    
    for i in range(number_of_communities):
        for j in range(number_of_communities):
            #if weight_between_communities[i][j] != 0:
            collapsed_AdjList[i].append((j,weight_between_communities[i][j]))
            
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
                B_matrix[i][eachnode[0]] += eachnode[1]

                
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


def change_in_U(number_of_nodes, i, old_comm, new_comm, community, B_matrix):
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
    
    if new_comm > old_comm:
        #print('we have new_comm > old_comm')
        #print('checking each node from the list - ', list(G.nodes()))
        for node in range(number_of_nodes):
            if node != i:
                #print('neighbour = ',node)
                if community[node] >= old_comm and community[node] < new_comm:
                    #print('we will add the B value for this neighbour')
                    delta = delta + B_matrix[node][i]
        #print('checking each node from the list - ', list(G.nodes())) 
        for node in range(number_of_nodes):
            if node!=i:
                #print('neighbour = ',node)
                if community[node] > old_comm and community[node] <= new_comm:
                    #print('we will subtract the B value for this neighbour')
                    delta = delta - B_matrix[i][node]
                
    elif new_comm < old_comm:
        #print('we have new_comm < old_comm')
        #print('checking each node from the list - ', list(G.nodes()))
        for node in range(number_of_nodes):
            if node != i:
                #print('neighbour = ',node)
                if community[node] >= new_comm and community[node] < old_comm:
                    #print('we will subtract the B value for this neighbour')
                    delta = delta - B_matrix[node][i]
                
        #print('checking each node from the list - ', list(G.nodes()))    
        for node in range(number_of_nodes):
            if node != i:
                #print('neighbour = ',node)
                if community[node] > new_comm and community[node] <= old_comm:
                    #print('we will add the B value for this neighbour')
                    delta = delta + B_matrix[i][node]
    #print(delta)            
    return delta


# In[14]:


def change_in_D(number_of_nodes, i, old_comm, new_comm, community, B_matrix):
    '''
    From the overall D, we can subtract the values which results from i leaving its old community and add the
    values which result from i joining its new community.
    
    We subtract all the *B values* for the edges going from i to all the nodes of community old_comm(i) and the edges
    entering i from the nodes of community old_comm(i), and we add all the *B values* for the edges going from i to 
    all the nodes of community new_comm(i) and the edges entering i from the nodes of community new_comm(i).
    '''
    delta = 0
    
    for node in range(number_of_nodes):
        if node!=i:
            if community[node] == old_comm:
                delta = delta - B_matrix[node][i]
            elif community[node] == new_comm:
                delta = delta + B_matrix[node][i]
            
    for node in range(number_of_nodes):
        if node!=i:
            if community[node] == old_comm:
                delta = delta - B_matrix[i][node]
            elif community[node] == new_comm:
                delta = delta + B_matrix[i][node]
            
    return delta


# In[15]:


def change_in_R(number_of_nodes, i, old_comm, new_comm, community, alpha, B_matrix):
    delta_U = change_in_U(number_of_nodes, i, old_comm, new_comm, community, B_matrix)
    
    delta_D = change_in_D(number_of_nodes, i, old_comm, new_comm, community, B_matrix)
    
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


def makeAMove(number_of_nodes, community, fr, alpha, c, old_R, B_matrix):
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
                    community[i] = r   #    move i to group r
                    
                    current_R = old_R + change_in_R(number_of_nodes, i, s, r, community, alpha, B_matrix)
                    #print("current_R in MAKE = ", current_R)
                    # calculate change in R for this new partition
                    current_R = round(current_R, 10)
                    best_R = round(best_R, 10)
                    # print(f'v[{i}] g[{int(s)}] --> g[{r}] : {thisR}')
                    if best_R == -np.inf: # because -np.inf cannot be converted to integer.
                        best_R    = current_R  #  best Rankularity so far
                        bestMove = [i,r]  #  the move that gets us there
                        
                    elif current_R > best_R:     # 
                    #elif current_R - best_R > 0.0000001:
                        best_R    = current_R  #  best Rankularity so far
                        bestMove = [i,r]  #  the move that gets us there
                        #print('best_R in MAKE is updated to ', best_R)
            community[i]= s   # put i back where we found it
                ##### do not modify below here #####  
                
    #print("Best to move node",bestMove[0],"from community", community[bestMove[0]], "to community",bestMove[1])
    #print('..which will make the R', bestR)
    return best_R,bestMove


# In[18]:


## Problem is in makeaMove, at the end all becomes 0,
## so size of set of community list becomes 1, so best_R stays at minus infinity, minus infinity is returned, and 
## bestMove has -1 community in it, current Rankularity =  -inf untill the next round phase is started, but then
## again the same thing happens.


# In[19]:


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
    for i in range(n):
        z[i] = int(rnd.randint(0,c-1))

    ##### do not modify below here #####

    return z


# In[20]:


def local_greedy_heuristic(AdjList, alpha_value, number_of_communities, rho, flag_model, degree_in, degree_out):
    T = 30
    n = len(AdjList)
    c = number_of_communities
    alpha = alpha_value
    
    
    community  = random_z(n,c) #[0,1,2,0,1,2,1,2,2,0,1]   # initial partition
    
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

    t  = 1  # counter for number of partitions considered, in this phase
    pc = 0  # counter for number of phases completed

    while True:
        # 3.1 visualization of this phase's initial partition
        #print('Phase no. = ',pc,"best_R =", bestR)
        #drawGz(G,current_partition)
        #b_rs = brs_matrix(G, current_partition, rho, flag_model)  # b_rs matrix to start with
        #current_R = calculate_rankularity(current_partition, b_rs, alpha)  # initial Rankularity score
        #current_R = best_R
        f = [0 for i in range(n)]  # no nodes frozen
        flag_updated = 0   # flag becomes 1 if we find at least one new move in the entire phase that increases our R.
        #print("\n \n new phase begun with best_R = ", best_R, "and partition ", current_partition)
        # loop over all the nodes in G, making greedy move for each
        #b_rs = brs_matrix(G, best_partition, rho, flag_model)  # b_rs matrix to start with
        #print(b_rs)
        #print("taken best_R = ", best_R)
        #best_R = calculate_rankularity(best_partition, b_rs, alpha)
        #print("after best_R = ", best_R)
        current_partition = copy.deepcopy(best_partition)
        ############
        current_R = best_R
        t = 0
        ############
        for j in range(n):
            
            #print(f'phase[{pc}] step {j}')
            #print('partition that goes into makeAMove function -', current_partition)
            #print("makeAMove sent with partiton ", current_partition, "and current Rankularity = ", current_R)
            #time11 = time.time()
            choice_R,choiceMove = makeAMove(n, current_partition, f, alpha, c, current_R, B_matrix)
            #print("choice_R returned by makeAMove = ", choice_R)
            choice_R = round(choice_R,10)
            best_R = round(best_R,10)
            i = choiceMove[0]    # move node i
            r = choiceMove[1]    # to community r
            f[i]    = 1          # freeze node i, so frozen list f is updated here 
            current_partition[i] = r          # make the greedy choice
            current_R = choice_R
            #print("current_changed partition = ", current_partition)

            # Remember, freezing a node and updating its community is mandatory in every iteration.
            Rankularity_values.append(current_R) # track Rankularity (even if it decreases)
            if choice_R > best_R:  # track best R and partition in the phase
                #print(choice_R, best_R, choiceMove)
                best_R = choice_R  # update the best Rankularity we have yet (among all phases)
                best_partition = copy.deepcopy(current_partition)  #
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
    return best_partition


# In[21]:


#needs to be fixed, probably the problem is not because of ER, rather the code is wrong


# In[22]:


# The Algo is working more or less correctly, plant a partition and check now. DONE - CORRECT!!
# Next, repeat the experiements done in paper with the fraction of correct nodes.

# Do further analysis on how the value of alpha (IMP), the number of community, 
# the value of noise p and other factors can affect the result.


# In[23]:


## So in order to know how Rankularity algo is performing, 
## it might be insightful to check the NMI between the ground truth and the algo's performance.


# In[24]:


## Now that Rankularity algo is working, NEXT STEP - 
## We construct networks in which the probability of each edge is chosen according 
## to the planted community structure of SBM with probability p and according to the 
## structurelessness with probability (1 − p), and then measure the accuracy of label 
## recovery as p is varied from 0 to 1.


# In[25]:


## First write a function to calculate fraction of correct nodes


# In[26]:


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
    
    print('fraction correct = ', fraction)
    return fraction


# In[27]:


def drawGz(G,z):
    # DO NOT MODIFY THIS FUNCTION
    # This function draws G with node labels from partition z
    #
    # input  : G is a networkx graph
    #        : z is a dictionary of group labels for G's nodes
    # output : none
    # 
    # WARNING: function is optimistic: assumes inputs are properly formatted

    plt.figure(figsize = (20,15))
    colors = ['#d61111','#11c6d6','#d67711','#11d646','#1b11d6','#d611cc'] # map node labels to colors (for the visualization)

    node_colors = []
    for i in G.nodes():
        node_colors.append(colors[int(z[int(i)-1])])
    nsize  = 700
    flabel = True

    if G.order() > 50:
        nsize  = 800
        flabel = True
        
    nx.draw_networkx(G, pos=nx.spring_layout(G), arrows=True, with_labels=flabel,node_size=nsize,width=2,node_color=node_colors, alpha = 1) # draw it pretty
    limits=plt.axis('off')                                      # turn off axes
    plt.show() 

    return


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + 2.5*pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.kamada_kawai_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def draw_trial(g, z):
    # to install networkx 2.0 compatible version of python-louvain use:
    # pip install -U git+https://github.com/taynaud/python-louvain.git@networkx2
    from community import community_louvain

    partition = {}
    for i in range(len(z)):
        partition[i] = z[i]
    pos = community_layout(g, partition)

    plt.figure(figsize = (20,15))
    colors = ['#d61111','#11c6d6','#d67711','#11d646','#1b11d6','#d611cc'] # map node labels to colors (for the visualization)

    node_colors = []
    for i in g.nodes():
        node_colors.append(colors[int(z[int(i)])])
        
    edge_colours = [i for i in range(g.number_of_edges())]
    nx.draw_networkx_nodes(g, pos, node_size=700,node_color=node_colors, alpha = 1)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize = 20, width=2,alpha = 0.5)

    #nx.draw(g, pos, node_color=list(partition.values()), with_labels=True)
    plt.show()
    return


# In[40]:


from collections import defaultdict

def run_Rankularity(realWorldNetwork, number_of_communities, flag_model):
    initial_time = time.time()
    
    def generate_graph(realWorldNetwork):
        # Read the real world network, nx.Graph object expected here.
        G = realWorldNetwork
        
        # Create the graph
        '''
        To store the graph, we will maintain a list of sets (we will use sets because search operation in sets are
        O(1)). This list will be of the size of total number of nodes nodes in graph and the sets at each index 
        will contain the set of neighbours of each node.
        '''
        total_number_of_nodes = G.number_of_nodes()
        AdjList = [set() for k in range(total_number_of_nodes)]
        degree_out = [0 for i in range(total_number_of_nodes)]
        degree_in = [0 for i in range(total_number_of_nodes)]
        
        for eachedge in G.edges():
            n1 = int(eachedge[0])
            n2 = int(eachedge[1])
            w = float(G.get_edge_data(eachedge[0],eachedge[1])['weight'])
            AdjList[n1].add((n2,w))
            degree_out[n1]+=w
            degree_in[n2]+=w
        
        
        potential_totalEdges = (total_number_of_nodes**2) - total_number_of_nodes
        sum_of_all_edgeWeights = 0
        for eachedge in G.edges():
            w = float(G.get_edge_data(eachedge[0],eachedge[1])['weight'])
            sum_of_all_edgeWeights += w
        
#         print("sum of all edgeweights = ", sum_of_all_edgeWeights)
#         check = 0
#         for node in range(len(AdjList)):
#             check += degree_out[node] 
#         print("check using deg_out and adjlist = ", check)

        rho = sum_of_all_edgeWeights/potential_totalEdges
#         print("Network density = ", rho)
#         for i in range(len(AdjList)):
#             print(i+1, "- ", end = " ")
#             if len(list((AdjList[i]))) == 0:
#                 print("{}")
#                 continue
#             print("{", end='')
#             for eachnode in AdjList[i]:
#                 print(eachnode+1, end = ", ")
#             print("\b",end="")
#             print("\b",end="")
#             print("}")
        
        return AdjList, rho, degree_in, degree_out
                    
    # G = nx.stochastic_block_model(sizes, probs, seed=0, directed=True)
    
    alpha_values = []
    alpha = 0
    while alpha <= 1:
        alpha_values.append(alpha)
        alpha = alpha + 0.1
        alpha = round(alpha,1)
        
    
    fractional_violations = [0 for i in range(11)]
    total_between_edges = [0 for i in range(11)]
    directionalities = [0 for i in range(11)]
    node_division = []
    run = 0
    alphawise_information = []
    print(alpha_values)
    while run < 1:
        if run%1 == 0:
            print("run = ", run)
        pos = 0
        alpha = 0
        AdjList, rho, degree_in, degree_out = generate_graph(realWorldNetwork)
        while alpha <= 1:
            #if run%1 == 0:
                #print("alpha = ", alpha)

            time1 = time.time()


            best_partition = local_greedy_heuristic(AdjList, alpha, number_of_communities, rho, flag_model, degree_in, degree_out)


            # Now using MVR -----
            collapsed_Graph = collapse_graph(AdjList, best_partition, number_of_communities)
            finalOrder, between_edges, fractional_final_violation, directionality = find_communityOrder(collapsed_Graph)
            #if run%10 == 0:
                #print("final order = ", finalOrder)
            total_between_edges[pos] += between_edges
            #print("between_edges = ", between_edges)
            #print("between_edges List = ", total_between_edges)
            fractional_violations[pos] += fractional_final_violation
            directionalities[pos] += directionality
            
            changed_ranks = []
            for i in range(number_of_communities):
                changed_ranks.append(finalOrder[i])
            final_answer = []
            for community in best_partition:
                final_answer.append(changed_ranks[community])

                
                
            node_count = defaultdict(int)
            for i in range(len(final_answer)):
                node_count[final_answer[i]] += 1
            if run == 0 and (alpha*10)%1 == 0:
                print("alpha = ", alpha)
                print("Collapsed Network = ", collapsed_Graph)
                print("Community order = ", finalOrder)
                print("Node counts = ", node_count)
                alphawise_information.append([alpha,collapsed_Graph,finalOrder,node_count])
                
            #if (run == 0 and alpha == 0.4) or (run == 0 and alpha == 0.2) or (run == 0 and alpha == 0.6) or (run == 0 and alpha == 0.8):
            inside_list = []
            inside_list.append(alpha)
            for c in range(number_of_communities):
                group_of_nodes = []
                for i in range(len(final_answer)):
                    if final_answer[i] == c:
                        group_of_nodes.append(i)
                inside_list.append(group_of_nodes)
            node_division.append(inside_list)
                
                
            pos += 1  
            #print(i, final_violations)
            time2 = time.time()
            
            alpha = alpha + 0.1
            alpha = round(alpha, 1)
        
        #print("final vio = ", final_violations)
        #print("fractional vio = ", fractional_violations)
        run+=1
     
    for i in range(len(total_between_edges)):
        total_between_edges[i] = total_between_edges[i]/1
        fractional_violations[i] = fractional_violations[i]/1
        directionalities[i] = directionalities[i]/1
        
    final_time = time.time()
    print("Total time taken =", (final_time - initial_time)/3600, "hours.")
    #print("After conversion")
    #print("final vio = ", final_violations)
    #print("fractional vio = ", fractional_violations)
    
    plt.figure(figsize = (15,6))
    plt.xticks(fontsize = 15 , color = 'black')
    plt.yticks(fontsize = 15 , color = 'black')
    plt.plot(alpha_values, total_between_edges, 'ro-', markersize = 3)
    plt.xlabel("Off diagonal factor, alpha" , fontsize = 15, color = 'black')
    plt.ylabel("Number of edges between communities" , fontsize = 15, color = 'black')
    plt.show()
    
    plt.figure(figsize = (15,7))
    plt.xticks(fontsize = 15 , color = 'black')
    plt.yticks(fontsize = 15 , color = 'black')
    plt.plot(alpha_values, fractional_violations, 'ro-' , markersize = 3)
    plt.xlabel("Off diagonal factor, alpha" , fontsize = 15 , color = 'black')
    plt.ylabel("Fraction of violations between communities" , fontsize = 15 , color = 'black')
    plt.show()
    
    plt.figure(figsize = (15,7))
    plt.xticks(fontsize = 15 , color = 'black')
    plt.yticks(fontsize = 15 , color = 'black')
    plt.plot(alpha_values, directionalities, 'ro-' , markersize = 3)
    plt.xlabel("Off diagonal factor, alpha" , fontsize = 15 , color = 'black')
    plt.ylabel("Directionality" , fontsize = 15 , color = 'black') # (#downEdges - #upEdges)/#inter-edges
    plt.show()
    
    return node_division, total_between_edges, fractional_violations, directionalities, alphawise_information


# In[1]:


def final_call(realWorldNetwork, number_of_communities):    
    rankularity_results = run_Rankularity(realWorldNetwork,number_of_communities,1)
    return rankularity_results

def convert_graph(G):
    print("number of nodes = ", G.number_of_nodes())
    print("number of edges = ", G.number_of_edges())
    converted_G = nx.DiGraph()
        
        
    hash_map = {} # A dictionary where key = node label in the original network G and value = the corresponding node number between 0 and n-1
    reverse_hash_map = {} # A dictionary where key = node number between 0 and n-1 and value = the corresponding node label in the original network G    
    
    nodelist = set()
    for eachedge in list(G.edges()):
        nodelist.add(eachedge[0])
        nodelist.add(eachedge[1])
    
    i = 0
    #print("number of nodes now = ", len(nodelist))
    for node in nodelist:
        hash_map[node] = i
        reverse_hash_map[i] = node
        converted_G.add_node(i)
        i+=1

    # Add edges to the new graph
    for eachedge in list(G.edges()):
        converted_G.add_edge(hash_map[eachedge[0]], hash_map[eachedge[1]], weight=G.get_edge_data(eachedge[0],eachedge[1])['weight'])

    return converted_G, hash_map, reverse_hash_map

def get_Rankularity_results(realWorldNetwork, number_of_communities):
    graph_and_hashes = convert_graph(realWorldNetwork)
    converted_network = graph_and_hashes[0]

    rankularity_results = final_call(converted_network, number_of_communities) ## Averaged over 50 iterations
    node_division = rankularity_results[0]
    alphawise_info = [rankularity_results[1], rankularity_results[2],rankularity_results[3],rankularity_results[4]]
    
    return graph_and_hashes,node_division,alphawise_info