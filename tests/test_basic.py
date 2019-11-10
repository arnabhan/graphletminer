# -*- coding: utf-8 -*-

import sys
import unittest
#sys.path.insert(0, "E:\Projects\Research\graphlet-miner")
from gminer.graphs import DocumentWordGraph, Graphlet



class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_graphlet_get_center_node(self):
        g = Graphlet('graphlet')
        self.assertEqual('graphlet', g.get_center_node())

    def test_graphlet_add_orbit(self):
        g = Graphlet('center')
        g.add_orbit()
        g.add_orbit()
        ''' Checking number of orbits: base orbit is the center node (orbit 0)'''
        print("Number of orbits =" + str(g.get_number_of_orbits()))
        self.assertEqual(3 , g.get_number_of_orbits())

    def test_graphlet_put_node_on_outer_orbit(self):
        g = Graphlet('center')
        g.add_orbit()
        g.put_node_on_outer_orbit('a')
        g.put_node_on_outer_orbit('b')
        g.put_node_on_outer_orbit('c')
        ''' Checking number of orbits: base orbit is the center node (orbit 0)'''
        node_list = g.get_nodes_on_outer_orbit()
        print("Nodes on the outer orbits =" + str(node_list))
        self.assertIs(True, 'a' in node_list and 'b' in node_list and 'c' in node_list and len(node_list) == 3)

    def test_graphlet_put_node_on_orbit(self):
        g = Graphlet('center')
        g.add_orbit()
        g.add_orbit()
        g.put_node_on_orbit('a',1)
        g.put_node_on_orbit('b',1)
        g.put_node_on_orbit('c',2)
        g.put_node_on_orbit('d',2)

        ''' Checking number of orbits: base orbit is the center node (orbit 0)'''
        o1_node_list = g.get_nodes_on_orbit(1)
        o2_node_list = g.get_nodes_on_orbit(2)
        print("Nodes on orbit 1 =" + str(o1_node_list))
        print("Nodes on orbit 2 =" + str(o2_node_list))
        self.assertIs(True, 'a' in o1_node_list and 'b' in o1_node_list and 'c' in o2_node_list and 'd' in o2_node_list and len(o1_node_list) == 2 and len(o2_node_list) == 2)

    def test_graphlet_get_all_nodes(self):
        g = Graphlet('center')
        g.add_orbit()
        g.put_node_on_outer_orbit('a')
        g.put_node_on_outer_orbit('b')
        g.put_node_on_outer_orbit('c')
        all_nodes = g.get_all_nodes()
        self.assertIs(True,'center' in all_nodes and 'a' in all_nodes and 'b' in all_nodes and 'c' in all_nodes  and len(all_nodes) == 4)


if __name__ == '__main__':
    unittest.main()

#unittest.main()