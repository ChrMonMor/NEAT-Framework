namespace NEAT
{
    public enum NodeType
    {
        INPUT = 0,
        HIDDEN = 1,
        OUTPUT = 2,
        BIAS = 3,
    }
    public class Node
    {

        internal int NodeId { get; set; }
        internal NodeType NodeType { get; set; }
        internal int NodeLayer { get; set; }
        internal float SumInput { get; set; }
        internal float SumOutput { get; set; }
        internal Node() { }

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
}
