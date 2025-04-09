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
            HiddenNodes = 0;
            OutputNodes = 1;
            ProcentConnection = 1f;
            SpeciesTarget = 5;
            ComputedThreshold = 4f;
            MutationRate = 1f;
            MutationChance = 0.8f;
            SpeciateCoefficient = new float[] {1f, 1f, 0.4f};

            var nodesArr = new int[] { InputNodes, HiddenNodes, OutputNodes };
            List<Node> bias = new List<Node>() { new Node(nodesArr.Length + 1, NodeType.BIAS, 1) };
            List<Brain> brains = Brain.RunInitialies(nodesArr, Pop_Size, bias, ProcentConnection);
            int n = 0;
            while (n < 1000)
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
                n++;
                Console.WriteLine(n);
            }
            Console.ReadLine();
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
        public static Brain BestEver = new Brain();
        public static List<Brain> NextGeneration(List<Brain> networks, int[] inHidOut, int popSize, List<Node> biasNodes = null, float connectProcent = 1f, float threshold = 99, float[] coefficients = null)
        {
            List<Brain> nextNetwork = Brain.RunInitialies(inHidOut, popSize, biasNodes, connectProcent);

            var fitest = networks.Max(y => y.Fitness);

            Brain BestInTest = networks.First(x => x.Fitness == fitest);

            if (BestInTest.Fitness > BestEver.Fitness) 
            {
                BestEver = BestInTest;
            }

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
                    Brain parrentA = SoftMaxSelect(last.Members, last.SumFitness);
                    Brain parrentB = SoftMaxSelect(last.Members, last.SumFitness);

                    Brain child = new Brain(parrentA, parrentB);

                    child.Mutate();

                    if (n < popSize)
                    {
                        nextNetwork[n++] = child;
                    } 
                    else
                    {
                        nextNetwork.Add(child);
                    }
                    
                }
            }

            if (Eltism)
            {
                if (nextNetwork.Count == 0) {
                    nextNetwork.Add(new Brain());
                }
                nextNetwork[0] = BestEver;
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
                sums += item.Fitness;
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
        public Species(int id, List<Brain> members = null, int allowedOffspring = 0, float fitness = 0f, float adjustedFitness = 0f, float sumFitness = 0f, float gensSinceImproved = 0f)
        {
            Id = id;
            Members = members ?? new List<Brain>();
            AllowedOffspring = allowedOffspring;
            Fitness = fitness;
            AdjustedFitness = adjustedFitness;
            SumFitness = sumFitness;
            GensSinceImproved = gensSinceImproved;
        }
        public int Id { get; set; }
        public List<Brain> Members { get; set; }
        public int AllowedOffspring { get; set; }
        public float Fitness { get; set; }
        public float AdjustedFitness { get; set; }
        public float SumFitness { get; set; }
        public float GensSinceImproved {  get; set; }
        public override string ToString() {
            return  "" + AdjustedFitness + ", " + AllowedOffspring + ", " + Members.Count;
        }
    }
    public static class Speciate
    {
        public static List<Species> Species = new List<Species>();
        public static float GlobalAdjustedFitness;
        public static int GenStagnationLimit = 15;
        public static List<Species> GenerationOffspring(List<Brain> networks, float threshold, float[] coefficient = null)
        {
            GlobalAdjustedFitness = 0;

            foreach (var s in Species) {

                var rep = s.Members[RNG.Ran(0, s.Members.Count)];
                s.SumFitness = 0;
                foreach (var item in networks) {
                    if (item.Species != -1) {
                        continue;
                    }
                    if (SpeciateComparisonCheck(rep, item, threshold, coefficient)) {
                        s.SumFitness += item.Fitness;
                    }
                }
                // SpeciesAvgFitness[n] /= networks.Count(x => x.Species == n);
            }

            int n = Species.Count;

            foreach (var net in networks)
            {
                if(net.Species != -1)
                {
                    continue;
                }
                Species.Add(new Species(n));
                var s = Species.Last();
                s.SumFitness = net.Fitness;
                net.Species = n;
                foreach (var item in networks)
                {
                    if (item.Species != -1)
                    {
                        continue;
                    }
                    if (SpeciateComparisonCheck(net, item, threshold, coefficient)) {
                        s.SumFitness += item.Fitness;
                    }
                }
                n++;
            }


            foreach (var brain in networks)
            {
                brain.AdjustedFitness = brain.Fitness / networks.Count(x => x.Species == brain.Species);
                Species.First(x => x.Id == brain.Species).AdjustedFitness += brain.AdjustedFitness;
            }


            float GlobalFitness = networks.Sum(x => x.AdjustedFitness) / networks.Count;

            foreach (var res in Species) 
            {
                res.Members = networks.Where(x => x.Species == res.Id).ToList();
                res.AdjustedFitness /= res.Members.Count; 
                res.AllowedOffspring = (int)Math.Round(res.AdjustedFitness / GlobalFitness * res.Members.Count, 0);
                res.SumFitness = res.Members.Sum(x => x.Fitness);

                if(res.Fitness < res.SumFitness / res.Members.Count) {
                    res.Fitness = res.SumFitness / res.Members.Count;
                    res.GensSinceImproved = 0;
                } 
                else 
                {
                    res.GensSinceImproved++;
                }
                if (res.GensSinceImproved >= GenStagnationLimit) {
                    res.AllowedOffspring = 0;
                }
            }

            for(int i = Species.Count-1; i >= 0; i--) 
            {
                if (Species[i].Members.Count <= 0) {
                    Species.Remove(Species[i]);
                }
            }

            GlobalAdjustedFitness = GlobalFitness;

            return Species;

        }
        public static bool SpeciateComparisonCheck(Brain netA, Brain netB, float threshold, float[] coefficient = null) {
            coefficient = coefficient ?? new[] { 1f, 1f, 1f };
            float excess = 0, disjoint = 0, avgWeight = 0;
            int n = Math.Max(netA.ArrConnections.Count, netB.ArrConnections.Count), avgCount = 0;
            int aMax = netA.ArrConnections.Max(x => x.Key), bMax = netB.ArrConnections.Max(x => x.Key);
            excess = (float)(netA.ArrConnections.Count(x => x.Key > bMax && x.Value.Enabled) + netB.ArrConnections.Count(x => x.Key > aMax && x.Value.Enabled)) / n * coefficient[0];
            disjoint = (float)(netA.ArrConnections.Count(x => x.Key < bMax && x.Value.Enabled && !netB.ArrConnections.ContainsKey(x.Key)) + netB.ArrConnections.Count(x => x.Key < aMax && x.Value.Enabled && !netA.ArrConnections.ContainsKey(x.Key))) / n * coefficient[1];
            // gives avg of connections of the same InnovationID
            foreach (var aConn in netA.ArrConnections) {
                if (aConn.Value.Enabled && netB.ArrConnections.ContainsKey(aConn.Key)) {
                    if (netB.ArrConnections[aConn.Key].Enabled)
                    {
                        avgWeight += Math.Abs(aConn.Value.ConnectionWeight - netB.ArrConnections[aConn.Key].ConnectionWeight);
                        avgCount++;
                    }
                }
            }

            avgWeight = (float)(avgWeight / avgCount) * coefficient[2];

            if (avgWeight == float.NaN)
            {
                avgWeight = 99;
            }

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
        public override string ToString()
        {
            return "" + NodeId + ", " + NodeType + ", " + NodeLayer;
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
        public override string ToString()
        {
            return "" + InnovationID + ", E:" + Enabled + ", w:" + ConnectionWeight;
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

        public Brain(Brain parentA, Brain parentB) {

            if (parentA.Fitness > parentB.Fitness)
            {
                foreach (var node in parentA.ArrNodes) 
                {
                    ArrNodes.Add(new Node(node.NodeId, node.NodeType, node.NodeLayer, node.SumInput, node.SumOutput));
                }

                foreach (var conn in parentA.ArrConnections) {
                    var c = conn.Value;
                    ArrConnections.Add(conn.Key, new Connection(c.InnovationID, c.InputNodeID, c.OutputNodeID, c.ConnectionWeight, c.Enabled, c.IsRecurrent));
                }
                Species = parentA.Species;
            }
            else
            {
                foreach (var node in parentB.ArrNodes) {
                    ArrNodes.Add(new Node(node.NodeId, node.NodeType, node.NodeLayer, node.SumInput, node.SumOutput));
                }

                foreach (var conn in parentB.ArrConnections) {
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
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count + 1, NodeLayer = layer, NodeType = NodeType.INPUT});
                    break;
                case NodeType.OUTPUT:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count + 1, NodeLayer = layer, NodeType = NodeType.OUTPUT});
                    break;
                case NodeType.HIDDEN:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count + 1, NodeLayer = layer, NodeType = NodeType.HIDDEN});
                    break;
                default:
                    ArrNodes.Add(new Node() { NodeId = ArrNodes.Count + 1, NodeLayer = layer, NodeType = NodeType.BIAS});
                    break;
            }
        }
        public bool AddConnection(Node input, Node output, int range = 20,bool enable = true, bool isRecurrent = false)
        {
            int id = int.Parse(input.NodeId + "00" + output.NodeId);

            if (ArrConnections.ContainsKey(id))
            {
                return false;
            }

            ArrConnections.Add(id, new Connection(id, input.NodeId, output.NodeId, RNG.RanDub() * (range*2) - range, enable, isRecurrent));
            return true;
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
                            } 
                            if (!ArrConnections[key].Enabled && RNG.RanDub() < mutationChanceDisabledConnection) {
                                ArrConnections[key].Enabled = true;
                                break;
                            }
                        } else {
                            AddConnection(a, b, range);
                            break;
                        }
                    }
                }
            }

            // Adding Nodes 
            if(RNG.RanDub() < mutaionChanceNodes) {
                Node a = ArrNodes[RNG.Ran(0, ArrNodes.Count)];
                foreach (var item in ArrConnections)
                {
                    if(item.Value.InputNodeID == a.NodeId && item.Value.Enabled)
                    {
                        Node c = ArrNodes[item.Value.OutputNodeID-1];
                        AddNode(a.NodeLayer+1, NodeType.HIDDEN);
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
        public void RebalanceNodeLayers() {
            // Reset all layers to uninitialized (-1)
            foreach (var node in ArrNodes) {
                if (node.NodeType == NodeType.INPUT || node.NodeType == NodeType.BIAS) {
                    node.NodeLayer = 1;
                } else {
                    node.NodeLayer = -1;
                }
            }

            bool changed = true;
            while (changed) {
                changed = false;
                foreach (var conn in ArrConnections.Values) {
                    if (!conn.Enabled) continue;

                    Node inputNode = ArrNodes.First(n => n.NodeId == conn.InputNodeID);
                    Node outputNode = ArrNodes.First(n => n.NodeId == conn.OutputNodeID);

                    int proposedLayer = inputNode.NodeLayer + 1;
                    if (inputNode.NodeLayer > 0 && (outputNode.NodeLayer == -1 || outputNode.NodeLayer < proposedLayer)) {
                        outputNode.NodeLayer = proposedLayer;
                        changed = true;
                    }
                }
            }

            // Ensure output nodes have the highest layer
            int maxLayer = ArrNodes.Max(n => n.NodeLayer);
            foreach (var node in ArrNodes) {
                if (node.NodeType == NodeType.OUTPUT) {
                    node.NodeLayer = maxLayer;
                }
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
