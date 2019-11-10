from graphviz import Digraph, nohtml

def create_dot_file(graphlet_pattern_support_map, output_file_path):
    '''
    Sample:
    0:likely_JJ/1:<FUNC_OR_STOP_WORD>|win_NN/2:<FUNC_OR_STOP_WORD>|pure_JJ|chocolate_NN     5
    orbits separated by '/'
    occupants separated by '|'
    orbitId and content separated by ':'
    '''
    dg = Digraph( node_attr={'shape': 'record'})
    dg.attr(compound='true')
    index = 0
    dg.attr(rankdir='RL')
    index = 0
    for k in graphlet_pattern_support_map.keys():
        graphlet = k # graphlet_pattern_support_map[index]
        support = list(set(graphlet_pattern_support_map[k]))[:3]
        with dg.subgraph(name='graphlet{0}'.format(index)) as s:
            s.attr(rank='same')
            orbits = graphlet.split('/')
            for i in range(len(orbits)):
                orbit_index, words =  orbits[i].split(':')
                words = words.replace('<','').replace('>','')
                if i == 0:
                    s.node('{0}{1}'.format(index,i), label='<f1>support={2}|{0}|{1}'.format(i,words,str(support)))
                else:
                    s.node('{0}{1}'.format(index,i), label='<f1>|{0}|{1}'.format(i,words))
            for i in range(len(orbits)-1):
                s.edge('{0}{1}:f1'.format(index,i), '{0}{1}:f1'.format(index,i+1), constraint='false' ) 
        index += 1 
    dg.render(output_file_path, view=True, renderer = 'cairo', formatter='cairo')

# def create_dot_graph(graphlet):
#     '''
#     Sample:
#     0:likely_JJ/1:<FUNC_OR_STOP_WORD>|win_NN/2:<FUNC_OR_STOP_WORD>|pure_JJ|chocolate_NN     5
#     orbits separated by '/'
#     occupants separated by '|'
#     orbitId and content separated by ':'
#     '''
#     dg = Digraph( node_attr={'shape': 'record', 'height':'.1'})
#     dg.attr(compound='true')
#     dg.attr(rankdir='RL')
#     orbits = graphlet.split('/')
#     for i in range(len(orbits)):
#         _, words =  orbits[i].split(':')
#         words = words.replace('<','').replace('>','')
#         dg.node('{0}'.format(i), label=nohtml('<f1>Orbit:{0}|{1}'.format(i,words)))
#     for i in range(len(orbits)-1):
#         dg.edge('{0}:f1'.format(i), '{0}:f1'.format(i+1), constraint='false' )  
#     return dg
    
def create_raw_dot_file(graphlet_list, output_file_path):
    '''
    Sample:
    0:likely_JJ/1:<FUNC_OR_STOP_WORD>|win_NN/2:<FUNC_OR_STOP_WORD>|pure_JJ|chocolate_NN     5
    orbits separated by '/'
    occupants separated by '|'
    orbitId and content separated by ':'
    '''
    dg = create_dot_graph(graphlet_list[0])
    dg.render(output_file_path, view=True, renderer = 'cairo', formatter='cairo')

#gp = ['0:likely_JJ/1:<FUNC_OR_STOP_WORD>|win_NN/2:<FUNC_OR_STOP_WORD>|pure_JJ|chocolate_NN','0:ico_NNP/1:ad_NN|<FUNC_OR_STOP_WORD>/2:<FUNC_OR_STOP_WORD>/3:officials_NNS']
#output_file='E:\\Projects\\Research\\Data\\analysis\\round-table.gv'
#create_raw_dot_file(gp, output_file)
#create_dot_file(gp,output_file)

