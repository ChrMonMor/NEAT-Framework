using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace NEAT
{
    public class Program
    {
        // Global Varibles
        public static int Pop_Size;
        public static int InputNodes;
        public static int HiddenNodes;
        public static int OutputNodes;
        public static float ProcentConnection; // goes between 0 and 1

        public static float[] SpeciateCoefficient;

        static void Main(string[] args)
        {
            Pop_Size = 50;
            InputNodes = 2;
            HiddenNodes = 0;
            OutputNodes = 1;
            ProcentConnection = 1f;
            SpeciateCoefficient = new float[] {1f, 1f, 0.4f};

            List<Brain> brains = new List<Brain>();

            for (int i = 0; i < Pop_Size; i++)
            {
                Brain brain = new Brain();
                var nodesArr = new int[] { InputNodes, HiddenNodes, OutputNodes };
                brain.Initialies(nodesArr, new List<Node>() { new Node(nodesArr.Length+1, NodeType.BIAS, 1) }, ProcentConnection);

                brain.LoadInputs(new float[] { 0, 0, 1 });
                brain.RunTheNetWork();
                brain.Fitness += 1 - brain.GetOutput().Average();

                brain.LoadInputs(new float[] { 0, 1, 1 });
                brain.RunTheNetWork();
                brain.Fitness += brain.GetOutput().Average();

                brain.LoadInputs(new float[] { 1, 0, 1 });
                brain.RunTheNetWork();
                brain.Fitness += brain.GetOutput().Average();

                brain.LoadInputs(new float[] { 1, 1, 1 });
                brain.RunTheNetWork();
                brain.Fitness += 1 - brain.GetOutput().Average();

                brains.Add(brain);
            }
            var a = Speciate.GenerationOffspring(brains, 6f, SpeciateCoefficient);
            brains = brains.OrderBy(x => x.Fitness).ToList();
            foreach (var item in brains)
            {
                Console.WriteLine(item.Fitness);
            }
            a = Speciate.GenerationOffspring(brains, 6f, SpeciateCoefficient);
            Console.WriteLine(a.Sum());
            Console.ReadLine();
        }
    }
    public class Species
    {
        public Species(int iD, List<Brain> members = null, List<Brain> offspring = null, float fitness = 0f, float adjustedFitness = 0f, float sumFitness = 0f, float gensSinceImproved = 0f)
        {
            ID = iD;
            Members = members ?? new List<Brain>();
            Offspring = offspring ?? new List<Brain>();
            Fitness = fitness;
            AdjustedFitness = adjustedFitness;
            SumFitness = sumFitness;
            GensSinceImproved = gensSinceImproved;
        }
        public int ID { get; set; }
        public List<Brain> Members { get; set; }
        public List<Brain> Offspring { get; set; }
        public float Fitness { get; set; }
        public float AdjustedFitness { get; set; }
        public float SumFitness { get; set; }
        public float GensSinceImproved {  get; set; }
    }
    public class Speciate
    {
        public static float GlobalFitness;
        public static Dictionary<int, float> SpeciesAvgFitness;
        public static List<int> Counts;
        public static int[] GenerationOffspring(List<Brain> networks, float threshold, float[] coefficient = null)
        {
            int n = 0;
            SpeciesAvgFitness = new Dictionary<int, float>();
            Counts = new List<int>();
            int a = RNG.Ran(0, networks.Count);
            // for starting of with a random index. 
            for (int i = a; i < networks.Count; i++)
            {
                if (networks[i].Species != -1)
                {
                    continue;
                }
                SpeciesAvgFitness.Add(n, 0f);
                Counts.Add(1);
                networks[i].Species = n;
                foreach (var item in networks)
                {
                    if (item.Species != -1)
                    {
                        continue;
                    }
                    if (SpeciateComparisonCheck(networks[i], item, threshold, coefficient))
                    {
                        Counts[n]++;
                    }
                }
                SpeciesAvgFitness[n] /= Counts[n];
                n++;
            }

            foreach (var net in networks)
            {
                if(net.Species != -1)
                {
                    continue;
                } 
                SpeciesAvgFitness.Add(n, 0f);
                Counts.Add(1);
                net.Species = n;
                foreach (var item in networks)
                {
                    if (item.Species != -1)
                    {
                        continue;
                    }
                    if(SpeciateComparisonCheck(net, item, threshold, coefficient))
                    {
                        Counts[n]++;
                    }
                }
                SpeciesAvgFitness[n] /= Counts[n];
                n++;
            }

            foreach (var brain in networks)
            {
                brain.AdjustedFitness = brain.Fitness / Counts[brain.Species];
                SpeciesAvgFitness[brain.Species] += brain.AdjustedFitness;
            }

            GlobalFitness = networks.Sum(x => x.AdjustedFitness) / networks.Count;

            int[] res = new int[n];

            for (int i = 0; i < n; i++)
            {
                res[i] = (int)Math.Round((SpeciesAvgFitness[i] / Counts[i]) / GlobalFitness * Counts[i],0);
            }

            return res;

        }
        public static void GlobalFitnessAdjusted(List<Brain> brains)
        {
            GlobalFitness = 0;
            foreach (var item in brains)
            {
                item.AdjustedFitness = item.Fitness / brains.Count(x => x.Species == item.Species);
                GlobalFitness += item.AdjustedFitness;
            }
            GlobalFitness /= brains.Count;
        }
        public static bool SpeciateComparisonCheck(Brain netA, Brain netB, float threshold, float[] coefficient = null) {
            coefficient = coefficient ?? new[] { 1f, 1f, 1f };
            float excess = 0, disjoint = 0, avgWeight = 0;
            int n = Math.Max(netA.ArrConnections.Count, netB.ArrConnections.Count), avgCount = 0;
            int aMax = netA.ArrConnections.Max(x => x.Key), bMax = netB.ArrConnections.Max(x => x.Key);
            excess = (float)(netA.ArrConnections.Count(x => x.Key > bMax) + netB.ArrConnections.Count(x => x.Key > aMax)) / n * coefficient[0];
            disjoint = (float)(netA.ArrConnections.Count(x => x.Key < bMax && !netB.ArrConnections.ContainsKey(x.Key)) + netB.ArrConnections.Count(x => x.Key < aMax && !netA.ArrConnections.ContainsKey(x.Key))) / n * coefficient[1];
            // gives avg of connections of the same InnovationID
            foreach (var aConn in netA.ArrConnections) {
                if (netB.ArrConnections.ContainsKey(aConn.Key)) {
                    avgWeight += Math.Abs(aConn.Value.ConnectionWeight + netB.ArrConnections[aConn.Key].ConnectionWeight);
                    avgCount++;
                }
            }

            avgWeight = (float)(avgWeight / avgCount) * coefficient[2];

            if(excess + disjoint + avgWeight <= threshold) {
                netB.Species = netA.Species;
                return true;
            }

            return false;
        }
        public static void AdjustedFitnessValue(Brain brain, int speciateCount)
        {
            brain.AdjustedFitness = brain.Fitness / speciateCount;
        }
    } 
    public enum NodeType
    {
        INPUT = 0,
        HIDDEN = 1,
        OUTPUT = 2,
        BIAS = 3,
    }
    public class Node
    {
        
        public int NodeId {  get; set; }
        public NodeType NodeType { get; set; }
        public int NodeLayer { get; set; }
        public float SumInput { get; set; }
        public float SumOutput { get; set; }
        public Node() { }

        public Node(int nodeId, NodeType nodeType, int nodeLayer, float sumInput = 0f, float sumOutput = 0f)
        {
            NodeId = nodeId;
            NodeType = nodeType;
            NodeLayer = nodeLayer;
            SumInput = sumInput;
            SumOutput = sumOutput;
        }
    }
    public class Connection
    {
        public int InnovationID { get; set; }
        public int InputNodeID { get; set; }
        public int OutputNodeID { get; set; }
        public float ConnectionWeight { get; set; }
        public bool Enabled { get; set; }
        public bool IsRecurrent { get; set; }
        public Connection() { }
        public Connection(int innovationID, int inputNodeID, int outputNodeID, float connectionWeight, bool enabled, bool isRecurrent)
        {
            InnovationID = innovationID;
            InputNodeID = inputNodeID;
            OutputNodeID = outputNodeID;
            ConnectionWeight = connectionWeight;
            Enabled = enabled;
            IsRecurrent = isRecurrent;
        }
    }
    public class Brain
    {
        public List<Node> ArrNodes = new List<Node>();
        public Dictionary<int, Connection> ArrConnections = new Dictionary<int, Connection>();
        public float Fitness = 0;
        public float AdjustedFitness = 0;
        public int Species = -1;
        public void Initialies(int[] inHidOut, List<Node> biasNodes = null, float connectProcent = 1f) 
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
                    if(connectProcent < RNG.RanDub())
                    {
                        continue;
                    }
                    if(iNode.NodeId == uNode.NodeId)
                    {
                        continue;
                    }
                    if (uNode.NodeLayer <= iNode.NodeLayer)
                    {
                        continue;
                    }
                    AddConnection(iNode, uNode);
                }
            }
        }
        public override string ToString()
        {
            return "" + Fitness + ", " + Species;
        }
        public void DrawNetwork()
        {

        }
        public void AddNode(int layer, NodeType type)
        {

            switch (type)
            {
                case NodeType.INPUT:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count, NodeLayer = layer, NodeType = NodeType.INPUT});
                    break;
                case NodeType.OUTPUT:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count, NodeLayer = layer, NodeType = NodeType.OUTPUT});
                    break;
                case NodeType.HIDDEN:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count, NodeLayer = layer, NodeType = NodeType.HIDDEN});
                    break;
                default:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count, NodeLayer = layer, NodeType = NodeType.BIAS});
                    break;
            }
        }
        public void AddConnection(Node input, Node output, int range = 20,bool enable = true, bool isRecurrent = false)
        {
            int id = int.Parse(input.NodeId + "00" + output.NodeId);

            if (ArrConnections.ContainsKey(id))
            {
                return;
            }

            ArrConnections.Add(id, new Connection(id, input.NodeId, output.NodeId, RNG.RanDub()* (range*2) - range, enable, isRecurrent));
        }
        public void Mutate()
        {

        }
        // We load only to the input nodes and they are only on layer 1
        // Since there are no activation fn, we also insert it as output sum
        // this is also a semi-run, for recurrent connections to the input-layer 
        public void LoadInputs(float[] input)
        {
            int i = 0;
            foreach (var node in ArrNodes) { 
                
                if(node.NodeLayer == 1)
                {
                    node.SumInput = input[i++];
                    foreach (var conn in ArrConnections)
                    {
                        if(conn.Value.InputNodeID == node.NodeId && conn.Value.IsRecurrent)
                        {
                            node.SumInput *= ArrNodes.First(x => x.NodeId == conn.Value.InputNodeID).SumOutput * conn.Value.ConnectionWeight;
                        }
                    }
                    node.SumOutput = node.SumInput;
                }
            }
        }
        public void RunTheNetWork()
        {
            int layer = 2;
            do
            {
                foreach (var node in ArrNodes)
                {
                    if(node.NodeLayer == layer)
                    {
                        node.SumInput = 0;
                        foreach (var conn in ArrConnections)
                        {
                            if(conn.Value.OutputNodeID == node.NodeId)
                            {
                                node.SumInput += ArrNodes.First(x => x.NodeId == conn.Value.InputNodeID).SumOutput * conn.Value.ConnectionWeight;
                            }
                            if(conn.Value.InputNodeID == node.NodeId && conn.Value.IsRecurrent)
                            {
                                node.SumInput += ArrNodes.First(x => x.NodeId == conn.Value.OutputNodeID).SumOutput * conn.Value.ConnectionWeight;
                            }
                        }
                        node.SumOutput = (float)(1 / (1 + Math.Exp(node.SumInput * -1.0f)));
                    }
                }
                layer++;
            }
            while (ArrNodes.Any(x => x.NodeLayer == layer));
        }
        public List<float> GetOutput(int layer)
        {
            return ArrNodes.Where(x => x.NodeLayer == layer).Select(x => x.SumOutput).ToList();
        }
        public List<float> GetOutput()
        {
            return ArrNodes.Where(x => x.NodeType == NodeType.OUTPUT).Select(x => x.SumOutput).ToList();
        }
    }
    static class RNG
    {
        private readonly static Random ran = new Random(); 
        public static float RanDub()
        {
            return (float)ran.NextDouble();
        }
        public static int Ran(int min, int max)
        {
            return ran.Next(min, max);
        }
    }
}
