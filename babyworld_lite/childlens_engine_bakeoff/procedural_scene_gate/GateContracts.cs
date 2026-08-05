#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using UnityEngine;

namespace ProceduralSceneGate
{
    public static class FrozenGate
    {
        public const int PhysicsHz = 240;
        public const int RenderHz = 30;
        public const int StepsPerFrame = 8;
        public const int Width = 1920;
        public const int Height = 1080;
        public const float DurationSeconds = 16f;
        public const string ConfigSchema = "embodied.procedural_scene_gate.v1";
        public const string TraceSchema = "embodied.episode_trace.v1";
        public static readonly string[] Digits = { "thumb", "index", "middle", "ring", "little" };
    }

    public enum TruthSource
    {
        Commanded,
        EngineObserved,
        PhysXMeasured,
        Derived,
        Unavailable
    }

    [Serializable]
    public struct PoseState
    {
        public Vector3 positionWorldM;
        public Quaternion rotationWorldXyzw;
        public Vector3 linearVelocityWorldMps;
        public Vector3 angularVelocityWorldRadps;
    }

    [Serializable]
    public sealed class ContactTruth
    {
        public int physicsStep;
        public string colliderA;
        public string colliderB;
        public Vector3 pointWorldM;
        public Vector3 normalWorld;
        public float separationM;
        public Vector3 relativeVelocityWorldMps;
        public Vector3 availableImpulseNs;
        public TruthSource provenance = TruthSource.PhysXMeasured;
    }

    public sealed class GateContext
    {
        public string EpisodeId;
        public string OutputRoot;
        public int Seed;
        public string RoomFamily;
        public string GarmentConfigurationId;
        public GameObject AuthorityRoot;
        public GameObject AvatarRoot;
        public Transform Torso;
        public Transform Neck;
        public Transform Head;
        public Transform LeftPalm;
        public Transform RightPalm;
        public readonly Dictionary<string, Transform[]> FingerSegments = new Dictionary<string, Transform[]>();
        public Rigidbody TargetBody;
        public Transform HeadCameraMount;
        public Camera HeadCamera;
        public Camera ExternalCamera;
        public int PhysicsStep;
        public int RenderFrame;
        public float TimeSeconds;
        public bool AnatomicalColliderVelocityDriveCommanded;
        public readonly List<ContactTruth> Contacts = new List<ContactTruth>();
        public readonly List<string> AssistanceLedger = new List<string>();
    }

    public interface IEmbodimentGarmentModule
    {
        void Build(GateContext context, string avatarAssetPath);
        void ApplyGarmentConfiguration(GateContext context, string configurationId);
        void MeasureRegistration(GateContext context, string reportPath);
    }

    public interface IFullBodyMotionModule
    {
        void Bind(GateContext context);
        void ApplyCommand(GateContext context, int physicsStep);
    }

    public interface ISceneCompilerModule
    {
        void Build(GateContext context, string furnitureAssetRoot);
    }

    public interface IPhysicsTruthModule
    {
        void Bind(GateContext context);
        void RecordAfterPhysicsStep(GateContext context);
        void Complete(GateContext context);
    }

    public interface IRegisteredCaptureModule
    {
        void Bind(GateContext context);
        void CaptureFrozenFrame(GateContext context);
        void Complete(GateContext context);
    }
}
#endif
