using System;
using System.Collections.Generic;
using System.Linq;

namespace NEAT
{
    public class Brain
    {
        public Brain() { }
        internal List<Node> ArrNodes = new List<Node>();
        internal Dictionary<int, Connection> ArrConnections = new Dictionary<int, Connection>();
        public float Fitness = 0;
        public float AdjustedFitness = 0;
        public int Species = -1;

        internal Brain(Brain parentA, Brain parentB)
        {

            if (parentA.Fitness > parentB.Fitness)
            {
                foreach (var node in parentA.ArrNodes)
                {
                    ArrNodes.Add(new Node(node.NodeId, node.NodeType, node.NodeLayer, node.SumInput, node.SumOutput));
                }

                foreach (var conn in parentA.ArrConnections)
                {
                    var c = conn.Value;
                    ArrConnections.Add(conn.Key, new Connection(c.InnovationID, c.InputNodeID, c.OutputNodeID, c.ConnectionWeight, c.Enabled, c.IsRecurrent));
                }
                Species = parentA.Species;
            }
            else
            {
                foreach (var node in parentB.ArrNodes)
                {
                    ArrNodes.Add(new Node(node.NodeId, node.NodeType, node.NodeLayer, node.SumInput, node.SumOutput));
                }

                foreach (var conn in parentB.ArrConnections)
                {
                    var c = conn.Value;
                    ArrConnections.Add(conn.Key, new Connection(c.InnovationID, c.InputNodeID, c.OutputNodeID, c.ConnectionWeight, c.Enabled, c.IsRecurrent));
                }
                Species = parentB.Species;
            }
            var temp = ArrConnections.Keys.ToList();
            foreach (var conn in temp)
            {
                if (parentA.ArrConnections.ContainsKey(conn) &&
                    parentB.ArrConnections.ContainsKey(conn))
                {
                    ArrConnections[conn].ConnectionWeight = RNG.RanDub() < 0.5 ? parentA.ArrConnections[conn].ConnectionWeight : parentB.ArrConnections[conn].ConnectionWeight;
                }
            }

            Fitness = 0;
            AdjustedFitness = 0;
        }
        public static List<Brain> RunInitialies(int[] inHidOut, int popSize, List<Node> biasNodes = null, float connectProcent = 1f)
        {
            List<Brain> brains = new List<Brain>();
            for (int i = 0; i < popSize; i++)
            {
                Brain brain = new Brain();
                brain.Initialies(inHidOut, biasNodes, connectProcent);
                brains.Add(brain);
            }
            return brains;
        }
        private void Initialies(int[] inHidOut, List<Node> biasNodes = null, float connectProcent = 1f, int connectionRange = 20)
        {
            int layer = 1;
            int type = 0;
            int j = 0;

            foreach (var n in inHidOut)
            {
                for (int i = 0; i < n; i++, j++)
                {
                    AddNode(layer, (NodeType)type);
                }
                type++;
                layer++;
            }
            if (biasNodes != null)
            {
                foreach (var node in biasNodes)
                {
                    AddNode(node.NodeLayer, NodeType.BIAS);
                }
            }

            foreach (var iNode in ArrNodes)
            {
                foreach (var uNode in ArrNodes)
                {
                    if (connectProcent < RNG.RanDub())
                    {
                        continue;
                    }
                    if (iNode.NodeId == uNode.NodeId)
                    {
                        continue;
                    }
                    if (uNode.NodeLayer <= iNode.NodeLayer)
                    {
                        continue;
                    }
                    AddConnection(iNode, uNode, range: connectionRange);
                }
            }
        }
        public override string ToString()
        {
            return "f: " + Fitness + ", s:" + Species + ", Cc:" + ArrConnections.Count + ", Nc:" + ArrNodes.Count;
        }
        public void DrawNetwork()
        {

        }
        private void AddNode(int layer, NodeType type)
        {

            switch (type)
            {
                case NodeType.INPUT:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count + 1, NodeLayer = layer, NodeType = NodeType.INPUT });
                    break;
                case NodeType.OUTPUT:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count + 1, NodeLayer = layer, NodeType = NodeType.OUTPUT });
                    break;
                case NodeType.HIDDEN:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count + 1, NodeLayer = layer, NodeType = NodeType.HIDDEN });
                    break;
                default:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count + 1, NodeLayer = layer, NodeType = NodeType.BIAS });
                    break;
            }
        }
        private bool AddConnection(Node input, Node output, int range = 20, bool enable = true, bool isRecurrent = false)
        {
            int id = int.Parse(input.NodeId + "00" + output.NodeId);

            if (ArrConnections.ContainsKey(id))
            {
                return false;
            }

            ArrConnections.Add(id, new Connection(id, input.NodeId, output.NodeId, RNG.RanDub() * (range * 2) - range, enable, isRecurrent));
            return true;
        }
        internal void Mutate(float mutateWeightsChance = 0.8f, float mutateWeightsRange = 0.2f, float mutateConnectionsChance = 0.05f, float mutateConnectionEnablingChance = 0.2f, bool recusionAllowed = false, int range = 20, float mutateNodesChance = 0.2f)
        {
            // Change Weigths 
            if (RNG.RanDub() < mutateWeightsChance)
            {
                foreach (var conn in ArrConnections)
                {
                    if (RNG.RanDub() < 0.9f)
                    {
                        float temp = conn.Value.ConnectionWeight * mutateWeightsRange;
                        conn.Value.ConnectionWeight += RNG.RanDub() * (temp * 2) - temp;
                    }
                    else
                    {
                        conn.Value.ConnectionWeight = RNG.RanDub() * (range * 2) - range;
                    }
                }
            }

            // Add Connection 
            if (RNG.RanDub() < mutateConnectionsChance)
            {
                for (int i = 0; i < 20; i++)
                {
                    Node a = ArrNodes[RNG.Ran(0, ArrNodes.Count)];
                    Node b = ArrNodes[RNG.Ran(0, ArrNodes.Count)];
                    if (a.NodeId != b.NodeId && a.NodeLayer < b.NodeLayer)
                    {
                        int key = int.Parse(a.NodeId + "00" + b.NodeId);
                        if (ArrConnections.ContainsKey(key))
                        {
                            if (recusionAllowed && ArrConnections[key].Enabled)
                            {
                                ArrConnections[key].IsRecurrent = !ArrConnections[key].IsRecurrent;
                                break;
                            }
                            if (!ArrConnections[key].Enabled && RNG.RanDub() < mutateConnectionEnablingChance)
                            {
                                ArrConnections[key].Enabled = true;
                                break;
                            }
                        }
                        else
                        {
                            AddConnection(a, b, range);
                            break;
                        }
                    }
                }
            }

            // Adding Nodes 
            if (RNG.RanDub() < mutateNodesChance)
            {
                Node a = ArrNodes[RNG.Ran(0, ArrNodes.Count)];
                foreach (var item in ArrConnections)
                {
                    if (item.Value.InputNodeID == a.NodeId && item.Value.Enabled)
                    {
                        Node c = ArrNodes[item.Value.OutputNodeID - 1];
                        AddNode(a.NodeLayer + 1, NodeType.HIDDEN);
                        Node b = ArrNodes.Last();
                        if (!AddConnection(a, b, enable: true, isRecurrent: item.Value.IsRecurrent))
                        {
                            continue;
                        }
                        if (!AddConnection(b, c, range))
                        {
                            continue;
                        }
                        ArrConnections.Last().Value.ConnectionWeight = item.Value.ConnectionWeight;
                        item.Value.Enabled = false;
                        break;
                    }
                }
                RebalanceNodeLayers();
            }

        }
        private void RebalanceNodeLayers()
        {
            foreach (var node in ArrNodes)
            {
                if (node.NodeType == NodeType.INPUT || node.NodeType == NodeType.BIAS)
                {
                    node.NodeLayer = 1;
                }
                else
                {
                    node.NodeLayer = -1;
                }
            }

            bool changed = true;
            while (changed)
            {
                changed = false;
                foreach (var conn in ArrConnections.Values)
                {
                    if (!conn.Enabled) continue;

                    Node inputNode = ArrNodes.First(n => n.NodeId == conn.InputNodeID);
                    Node outputNode = ArrNodes.First(n => n.NodeId == conn.OutputNodeID);

                    int proposedLayer = inputNode.NodeLayer + 1;
                    if (inputNode.NodeLayer > 0 && (outputNode.NodeLayer == -1 || outputNode.NodeLayer < proposedLayer))
                    {
                        outputNode.NodeLayer = proposedLayer;
                        changed = true;
                    }
                }
            }

            int maxLayer = ArrNodes.Max(n => n.NodeLayer);
            foreach (var node in ArrNodes)
            {
                if (node.NodeType == NodeType.OUTPUT)
                {
                    node.NodeLayer = maxLayer;
                }
            }
        }
        // We load only to the input nodes and they are only on layer 1
        // Since there are no activation fn, we also insert it as output sum
        // this is also a semi-run, for recurrent connections to the input-layer 
        private void LoadInputs(float[] input)
        {
            int i = 0;
            foreach (var node in ArrNodes)
            {

                if (node.NodeLayer == 1)
                {
                    node.SumInput = input[i++];
                    foreach (var conn in ArrConnections)
                    {
                        if (conn.Value.InputNodeID == node.NodeId && conn.Value.IsRecurrent)
                        {
                            node.SumInput *= ArrNodes.First(x => x.NodeId == conn.Value.InputNodeID).SumOutput * conn.Value.ConnectionWeight;
                        }
                    }
                    node.SumOutput = node.SumInput;
                }
            }
        }
        private void RunTheNetWork()
        {
            int layer = 2;
            int maxLayer = ArrNodes.Max(n => n.NodeLayer);

            while (layer <= maxLayer)
            {
                foreach (var node in ArrNodes)
                {
                    if (node.NodeLayer == layer)
                    {
                        node.SumInput = 0;

                        foreach (var conn in ArrConnections.Values)
                        {
                            if (!conn.Enabled) continue;

                            if (conn.OutputNodeID == node.NodeId)
                            {
                                Node inputNode = ArrNodes.First(x => x.NodeId == conn.InputNodeID);
                                if (inputNode.NodeLayer != layer) node.SumInput += inputNode.SumOutput * conn.ConnectionWeight;
                            }

                            if (conn.InputNodeID == node.NodeId && conn.IsRecurrent)
                            {
                                Node outputNode = ArrNodes.First(x => x.NodeId == conn.OutputNodeID);
                                if (outputNode.NodeLayer != layer) node.SumInput += outputNode.SumOutput * conn.ConnectionWeight;
                            }
                        }
                        node.SumOutput = Activation(node.SumInput);
                    }
                }
                layer++;
            }
        }

        // Activation function
        private float Activation(float x)
        {
            return 1f / (1f + (float)Math.Exp(-x));
        }

        private List<float> GetOutput(int layer)
        {
            return ArrNodes.Where(x => x.NodeLayer == layer).Select(x => x.SumOutput).ToList();
        }
        private List<float> GetOutput()
        {
            return ArrNodes.Where(x => x.NodeType == NodeType.OUTPUT).Select(x => x.SumOutput).ToList();
        }
        public List<float> Run(float[] input)
        {
            {
                LoadInputs(input);
                RunTheNetWork();
                return GetOutput();
            }
        }
    }
}
