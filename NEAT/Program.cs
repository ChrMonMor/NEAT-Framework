using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading;
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
        public static int SpeciesTarget;
        public static float ComputedThreshold;
        public static float MutationRate;
        public static float MutationChance;

        public static float[] SpeciateCoefficient;

        static void Main(string[] args)
        {
            Pop_Size = 50;
            InputNodes = 2;
            HiddenNodes = 1;
            OutputNodes = 1;
            ProcentConnection = 1f;
            SpeciesTarget = 5;
            ComputedThreshold = 30f;
            MutationRate = 1f;
            MutationChance = 0.8f;
            SpeciateCoefficient = new float[] {1f, 1f, 0.4f};

            var nodesArr = new int[] { InputNodes, HiddenNodes, OutputNodes };
            List<Node> bias = new List<Node>() { new Node(nodesArr.Length + 1, NodeType.BIAS, 1) };
            List<Brain> brains = Brain.RunInitialies(nodesArr, Pop_Size, bias, ProcentConnection);

            while (true)
            {
                TestRun(brains);

                brains = Crossover.NextGeneration(brains, nodesArr, Pop_Size, bias, ProcentConnection, ComputedThreshold, SpeciateCoefficient);

                if (Speciate.Species.Count > SpeciesTarget)
                {
                    ComputedThreshold += 0.5f ;
                } 
                else
                {
                    ComputedThreshold -= 0.5f;
                }

                // this is to be deleted as I will Only do this for readabilit
                Console.WriteLine(brains.First(x => x.Fitness == brains.Max(y => y.Fitness)).ToString());
                Console.WriteLine(Speciate.Species.Sum(x => x.Members.Count));
                Console.WriteLine(Speciate.GlobalAdjustedFitness);
                Console.WriteLine(ComputedThreshold);
                Console.WriteLine(Speciate.Species.Max(x => x.GensSinceImproved));
                Console.Clear();
            }
        }
        public static void TestRun(List<Brain> brains)
        {
            for (int i = 0; i < Pop_Size; i++)
            {
                brains[i].LoadInputs(new float[] { 0, 0, 1 });
                brains[i].RunTheNetWork();
                brains[i].Fitness = 1 - brains[i].GetOutput().Average();

                brains[i].LoadInputs(new float[] { 0, 1, 1 });
                brains[i].RunTheNetWork();
                brains[i].Fitness += brains[i].GetOutput().Average();

                brains[i].LoadInputs(new float[] { 1, 0, 1 });
                brains[i].RunTheNetWork();
                brains[i].Fitness += brains[i].GetOutput().Average();

                brains[i].LoadInputs(new float[] { 1, 1, 1 });
                brains[i].RunTheNetWork();
                brains[i].Fitness += 1 - brains[i].GetOutput().Average();
            }
        }
    }
    public static class Crossover
    {
        public static bool Eltism = true;
        public static List<Brain> NextGeneration(List<Brain> networks, int[] inHidOut, int popSize, List<Node> biasNodes = null, float connectProcent = 1f, float threshold = 99, float[] coefficients = null)
        {
            List<Brain> nextNetwork = Brain.RunInitialies(inHidOut, popSize, biasNodes, connectProcent);

            var fitest = networks.Max(y => y.Fitness);
            Brain BestInTest = networks.First(x => x.Fitness == fitest);

            foreach (var net in networks)
            {
                net.Species = -1;
            }

            List<Species> speciesList = Speciate.GenerationOffspring(networks, threshold, coefficients);

            int n = 0;
            foreach (var last in speciesList)
            {
                for (int i = 0; i < last.AllowedOffspring; i++)
                {
                    Brain parrentA = SoftMaxSelect(last.Members, last.SumAdjustedFitness);
                    Brain parrentB = SoftMaxSelect(last.Members, last.SumAdjustedFitness);

                    Brain child = new Brain(parrentA, parrentB);
                                        
                    nextNetwork[n++] = (child);
                }
            }

            if (Eltism)
            {
                if (nextNetwork.Count == 0) {
                    nextNetwork.Add(new Brain());
                }
                nextNetwork[0] = BestInTest;
            }

            
            return nextNetwork;

        }
        public static void EltiOnOff()
        {
            Eltism = !Eltism;
        }
        public static Brain SoftMaxSelect(List<Brain> values, float maxValue)
        {
            float selected = RNG.RanDub() * (maxValue * 2) + 0;
            float sums = 0;

            foreach (var item in values)
            {
                sums += item.AdjustedFitness;
                if (sums > selected)
                {
                    return item;
                }
            }

            return values.Last();
        }
    }
    public class Species
    {
        public Species(int id, List<Brain> members = null, int allowedOffspring = 0, float fitness = 0f, float adjustedFitness = 0f, float sumAdjustedFitness = 0f, float gensSinceImproved = 0f)
        {
            Id = id;
            Members = members ?? new List<Brain>();
            AllowedOffspring = allowedOffspring;
            Fitness = fitness;
            AdjustedFitness = adjustedFitness;
            SumAdjustedFitness = sumAdjustedFitness;
            GensSinceImproved = gensSinceImproved;
        }
        public int Id { get; set; }
        public List<Brain> Members { get; set; }
        public int AllowedOffspring { get; set; }
        public float Fitness { get; set; }
        public float AdjustedFitness { get; set; }
        public float SumAdjustedFitness { get; set; }
        public float GensSinceImproved {  get; set; }
        public override string ToString() {
            return  "" + AdjustedFitness + ", " + AllowedOffspring + ", " + Members.Count;
        }
    }
    public static class Speciate
    {
        public static List<Species> Species { get; set; }
        public static float GlobalAdjustedFitness;
        public static int GenStagnationLimit = 15;
        public static List<Species> GenerationOffspring(List<Brain> networks, float threshold, float[] coefficient = null)
        {
            GlobalAdjustedFitness = 0;
            int n = 0;
            Dictionary<int, float> SpeciesAvgFitness = new Dictionary<int, float>();
            int a = RNG.Ran(0, networks.Count);
            // for starting of with a random index. 
            for (int i = a; i < networks.Count; i++)
            {
                if (networks[i].Species != -1)
                {
                    continue;
                }
                SpeciesAvgFitness.Add(n, 0f);
                networks[i].Species = n;
                foreach (var item in networks)
                {
                    if (item.Species != -1)
                    {
                        continue;
                    }
                    SpeciateComparisonCheck(networks[i], item, threshold, coefficient);
                }
                SpeciesAvgFitness[n] /= networks.Count(x => x.Species == n);
                n++;
            }

            foreach (var net in networks)
            {
                if(net.Species != -1)
                {
                    continue;
                } 
                SpeciesAvgFitness.Add(n, 0f);
                net.Species = n;
                foreach (var item in networks)
                {
                    if (item.Species != -1)
                    {
                        continue;
                    }
                    SpeciateComparisonCheck(net, item, threshold, coefficient);
                }
                SpeciesAvgFitness[n] /= networks.Count(x => x.Species == n);
                n++;
            }

            foreach (var brain in networks)
            {
                brain.AdjustedFitness = brain.Fitness / networks.Count(x => x.Species == brain.Species);
                SpeciesAvgFitness[brain.Species] += brain.AdjustedFitness;
            }

            float GlobalFitness = networks.Sum(x => x.AdjustedFitness) / networks.Count;

            List<Species> res = new List<Species>();

            Dictionary<Brain, float> represents = new Dictionary<Brain, float>();

            if(Species != null)
            {
                foreach (var rep in Species)
                {
                    var temp = Crossover.SoftMaxSelect(rep.Members, rep.SumAdjustedFitness);
                    represents.Add(temp, rep.GensSinceImproved);
                }
            }

            for (int i = 0; i < n; i++)
            {
                res.Add(new Species(i));
                res[i].AdjustedFitness = (SpeciesAvgFitness[i] / networks.Count(x => x.Species == i)); 
                res[i].AllowedOffspring = (int)Math.Round(res[i].AdjustedFitness / GlobalFitness * networks.Count(x => x.Species == i),0);
                res[i].Members = networks.Where(x => x.Species == i).ToList();
                res[i].SumAdjustedFitness = res[i].Members.Sum(x => x.AdjustedFitness);
                res[i].Fitness = res[i].Members.Sum(x => x.Fitness) / res[i].Members.Count;
                res[i].GensSinceImproved = 0;
                foreach (var rep in represents)
                {
                    if (res[i].Members.Any(x => x.Fitness == rep.Key.Fitness && res[i].Id == rep.Key.Species))
                    {
                        res[i].GensSinceImproved = rep.Value + 1;
                    }
                }
                if(res[i].GensSinceImproved >= GenStagnationLimit)
                {
                    res[i].AllowedOffspring = 0;
                }
                
            }

            GlobalAdjustedFitness = GlobalFitness;
            Species = res;

            return res;

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
                    avgWeight += Math.Abs(aConn.Value.ConnectionWeight - netB.ArrConnections[aConn.Key].ConnectionWeight);
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
        public Brain() { }
        public List<Node> ArrNodes = new List<Node>();
        public Dictionary<int, Connection> ArrConnections = new Dictionary<int, Connection>();
        public float Fitness = 0;
        public float AdjustedFitness = 0;
        public int Species = -1;

        public Brain(Brain parentA, Brain parentB = null) {

            if (parentA.Fitness < parentB.Fitness)
            {
                this.ArrNodes = parentA.ArrNodes;
                this.ArrConnections = parentA.ArrConnections;
                this.Species = parentA.Species;
            }
            else
            {
                this.ArrNodes = parentB.ArrNodes;
                this.ArrConnections = parentB.ArrConnections;
                this.Species = parentB.Species;
            }
            var temp = this.ArrConnections.Keys.ToList();
            foreach (var conn in temp)
            {
                if (parentA.ArrConnections.ContainsKey(conn) &&
                    parentB.ArrConnections.ContainsKey(conn))
                {
                    this.ArrConnections[conn] = RNG.RanDub() < 0.5 ? parentA.ArrConnections[conn] : parentB.ArrConnections[conn]; ;
                }
            }

            this.Fitness = 0;
            this.AdjustedFitness = 0;
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
        public void Initialies(int[] inHidOut, List<Node> biasNodes = null, float connectProcent = 1f, int connectionRange = 20) 
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
                    AddConnection(iNode, uNode, range: connectionRange);
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

            ArrConnections.Add(id, new Connection(id, input.NodeId, output.NodeId, RNG.RanDub() * (range*2) - range, enable, isRecurrent));
        }
        public void Mutate(float mutationChanceWeight = 0.8f, float mutationRateWeight = 0.2f, float mutationChanceConnection = 0.05f, float mutationChanceDisabledConnection = 0.2f, bool recusionAllowed = false, int range = 20, float mutaionChanceNodes = 0.2f)
        {
            // Change Weigths 
            if(RNG.RanDub() < mutationChanceWeight) {
                foreach (var conn in ArrConnections) {
                    if(RNG.RanDub() < 0.9f) {
                        float temp = conn.Value.ConnectionWeight * mutationRateWeight;
                        conn.Value.ConnectionWeight += RNG.RanDub() * (temp * 2) - temp;
                    } else {
                        conn.Value.ConnectionWeight = RNG.RanDub() * (range * 2) - range;
                    }
                }
            }

            // Add Connection 
            if(RNG.RanDub() < mutationChanceConnection) {
                for (int i = 0; i < 20; i++) {
                    Node a = ArrNodes[RNG.Ran(0, ArrNodes.Count)];
                    Node b = ArrNodes[RNG.Ran(0, ArrNodes.Count)];
                    if(a.NodeId != b.NodeId && a.NodeLayer < b.NodeLayer ) {
                        int key = int.Parse(a.NodeId + "00" + b.NodeId);
                        if (ArrConnections.ContainsKey(key)) {
                            if (recusionAllowed && ArrConnections[key].Enabled) {
                                ArrConnections[key].IsRecurrent = !ArrConnections[key].IsRecurrent;
                                break;
                            } else if (!ArrConnections[key].Enabled && RNG.RanDub() < mutationChanceDisabledConnection) {
                                ArrConnections[key].Enabled = true;
                                break;
                            }
                        } else {
                            AddConnection(a, b, range: range);
                            break;
                        }
                    }
                }
            }

            // Adding Nodes 
            if(RNG.RanDub() < mutaionChanceNodes) {

            }
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
