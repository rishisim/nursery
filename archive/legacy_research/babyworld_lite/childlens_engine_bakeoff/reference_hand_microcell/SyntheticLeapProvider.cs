using System.Collections.Generic;
using Leap;
using UnityEngine;

namespace EmbodiedReferenceHand
{
    public sealed class SyntheticLeapProvider : LeapProvider
    {
        private Frame current = new Frame();

        public override Frame CurrentFrame => current;
        public override Frame CurrentFixedFrame => current;

        public void EmitFixedFrame(Hand hand)
        {
            current = new Frame(hand.FrameId, hand.FrameId * 1000000L / 240L, 240f,
                new List<Hand> { hand });
            DispatchFixedFrameEvent(current);
        }
    }
}
