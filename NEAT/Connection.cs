namespace NEAT
{
    internal class Connection
    {
        internal int InnovationID { get; set; }
        internal int InputNodeID { get; set; }
        internal int OutputNodeID { get; set; }
        internal float ConnectionWeight { get; set; }
        internal bool Enabled { get; set; }
        internal bool IsRecurrent { get; set; }
        internal Connection() { }
        internal Connection(int innovationID, int inputNodeID, int outputNodeID, float connectionWeight, bool enabled, bool isRecurrent)
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
}
