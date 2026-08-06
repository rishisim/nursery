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
        public const float DurationSeconds = 24f;
        public const float RightForceOppositionSeconds = 0.30f;
        public const float LeftForceSupportSeconds = 0.25f;
        public const float FingerObjectPenetrationMaxM = 0.002f;
        public const float SupportPenetrationMaxM = 0.0015f;
        public const string ConfigSchema = "embodied.integrated_dexterous_scene_gate.v2";
        public const string TraceSchema = "embodied.episode_trace.v2";
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
        public string CellId;
        public string TargetId;
        public string DestinationId;
        public string ContactStrategy;
        public string FinalGazeZone;
        public string CompiledContractPath;
        public string CompiledContractSha256;
        public string RobustnessVariant;
        public string ReplayTracePath;
        public float TargetMidpointReachM;
        public float TargetReachBandMinM;
        public float TargetReachBandMaxM;
        public float TargetLateralBiasM;
        public Vector3 TargetDimensionsM;
        public float TargetMassKg;
        public float TargetStaticFriction;
        public float TargetDynamicFriction;
        public string TargetGeometry;
        public int TargetSemanticId;
        public int TargetInstanceId;
        public Vector3 SceneEnvelopeM;
        public string SceneMaterialVariant;
        public string[] SceneZoneIds;
        public string[] ExpectedSceneAssetIds;
        public SceneInstanceAuthority[] ExpectedSceneInstances;
        public SceneSupportAuthority[] ExpectedSupportRelations;
        public bool ExpectedTargetVisibleAtRequiredEvents;
        public string ExpectedFinalGazeZone;
        public float SceneStabilizationSeconds;
        public bool RequireNoVisiblePrimitiveFurniture;
        public int MinimumContextualObjects;
        public GameObject AuthorityRoot;
        public GameObject AvatarRoot;
        public Transform Torso;
        public Transform Neck;
        public Transform Head;
        public Transform LeftPalm;
        public Transform RightPalm;
        public readonly Dictionary<string, Transform[]> FingerSegments = new Dictionary<string, Transform[]>();
        public readonly Dictionary<string, Transform> BodySegments = new Dictionary<string, Transform>();
        public readonly Dictionary<string, Rigidbody> FingerBodies = new Dictionary<string, Rigidbody>();
        public readonly Dictionary<string, ConfigurableJoint> FingerJoints = new Dictionary<string, ConfigurableJoint>();
        public readonly Dictionary<string, Transform> FingerAuthorityBones = new Dictionary<string, Transform>();
        public readonly HashSet<Collider> AvatarColliders = new HashSet<Collider>();
        public readonly Dictionary<string, Transform> Destinations = new Dictionary<string, Transform>();
        public readonly Dictionary<string, Vector2> ActivityPhasesSeconds = new Dictionary<string, Vector2>();
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
        public readonly List<AuthorityLedgerEntry> RecoveryLedger = new List<AuthorityLedgerEntry>();
        public readonly AuthorityAuditState AuthorityAudit = new AuthorityAuditState();
    }

    [Serializable]
    public sealed class SceneInstanceAuthority
    {
        public string PersistentId;
        public string AssetId;
        public Vector3 AssetDimensionsM;
        public string SemanticClass;
        public string CollisionSource;
        public bool Interactive;
        public float MassKg;
        public float StaticFriction;
        public float DynamicFriction;
    }

    [Serializable]
    public sealed class SceneSupportAuthority
    {
        public string ChildId;
        public string SupportId;
        public string DestinationId;
    }

    [Serializable]
    public sealed class AuthorityLedgerEntry
    {
        public int physicsStep;
        public string category;
        public string actor;
        public string target;
        public string detail;
    }

    [Serializable]
    public sealed class AuthorityAuditState
    {
        public int targetPoseWriteCounter;
        public int targetVelocityWriteCounter;
        public int targetForceCounter;
        public int targetTorqueCounter;
        public int targetJointCounter;
        public int targetParentingCounter;
        public int targetKinematicChangeCounter;
        public int recoveryCounter;
        public string sourceAuditSha256 = "UNSEALED";
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
        void SynchronizeCompletedPhysicsState(GateContext context, int physicsStep);
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
