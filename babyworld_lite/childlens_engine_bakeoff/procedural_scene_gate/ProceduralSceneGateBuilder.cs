#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace ProceduralSceneGate
{
    public static class ProceduralSceneGateBuilder
    {
        private const float Dt = 1f / FrozenGate.PhysicsHz;
        private const int TotalSteps = (int)(FrozenGate.DurationSeconds * FrozenGate.PhysicsHz);

        [MenuItem("BabyWorld/Run Procedural Clothed Scene Gate")]
        public static void Run()
        {
            string output = RequiredEnvironment("PROCEDURAL_GATE_OUTPUT");
            Directory.CreateDirectory(output);
            ClearScene();
            ConfigurePhysics();
            var context = new GateContext
            {
                EpisodeId = RequiredEnvironment("PROCEDURAL_GATE_EPISODE_ID"),
                OutputRoot = output,
                Seed = RequiredInt("PROCEDURAL_GATE_SEED"),
                RoomFamily = RequiredEnvironment("PROCEDURAL_GATE_ROOM_FAMILY"),
                GarmentConfigurationId = RequiredEnvironment("PROCEDURAL_GATE_GARMENT_CONFIG"),
                AuthorityRoot = new GameObject("AUTHORITATIVE_PHYSICS_CLOCKED_STATE")
            };
            string stage = Environment.GetEnvironmentVariable("PROCEDURAL_GATE_STAGE") ?? "integrated";
            ISceneCompilerModule scene = new ProceduralSceneCompiler();
            var embodiment = new EmbodimentGarments();
            IFullBodyMotionModule motion = new FullBodyBimanualMotion();
            IPhysicsTruthModule truth = new PhysicsTruthRecorder();
            IRegisteredCaptureModule capture = new RegisteredCapture();

            embodiment.Build(context, "Assets/Avatar/child.fbx");
            embodiment.ApplyGarmentConfiguration(context, context.GarmentConfigurationId);
            scene.Build(context, "Assets/Furniture");
            if ((stage == "garment_sweep" || stage == "motion_camera") && context.TargetBody)
                context.TargetBody.gameObject.SetActive(false);
            motion.Bind(context);
            capture.Bind(context);
            truth.Bind(context);
            Physics.SyncTransforms();

            for (int step = 0; step < TotalSteps; step++)
            {
                context.PhysicsStep = step;
                context.TimeSeconds = step * Dt;
                motion.ApplyCommand(context, step);
                embodiment.UpdateRegisteredCollidersBeforePhysics(context);
                Physics.SyncTransforms();
                Physics.Simulate(Dt);
                Physics.SyncTransforms();
                context.TimeSeconds = (step + 1) * Dt;
                embodiment.SampleRegistrationAtPhysicsStep(context);
                context.RenderFrame = step / FrozenGate.StepsPerFrame;
                truth.RecordAfterPhysicsStep(context);
                if ((step + 1) % FrozenGate.StepsPerFrame == 0)
                {
                    capture.CaptureFrozenFrame(context);
                }
            }

            truth.Complete(context);
            capture.Complete(context);
            embodiment.MeasureRegistration(context, Path.Combine(output, "registration_report.json"));
            WriteAuthorityReceipt(context, stage);
            AssetDatabase.SaveAssets();
            EditorApplication.Exit(context.AssistanceLedger.Count == 0 ? 0 : 3);
        }

        private static void ConfigurePhysics()
        {
            Physics.simulationMode = SimulationMode.Script;
            Physics.gravity = new Vector3(0f, -9.81f, 0f);
            Physics.defaultSolverIterations = 24;
            Physics.defaultSolverVelocityIterations = 12;
            Physics.reuseCollisionCallbacks = false;
        }

        private static void ClearScene()
        {
            foreach (GameObject gameObject in UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None))
                UnityEngine.Object.DestroyImmediate(gameObject);
        }

        private static void WriteAuthorityReceipt(GateContext context, string stage)
        {
            var receipt = new AuthorityReceipt
            {
                schema = "embodied.single_authority_receipt.v1",
                episode_id = context.EpisodeId,
                stage = stage,
                physics_hz = FrozenGate.PhysicsHz,
                render_hz = FrozenGate.RenderHz,
                steps_per_render_frame = FrozenGate.StepsPerFrame,
                physics_steps = TotalSteps,
                render_frames = TotalSteps / FrozenGate.StepsPerFrame,
                authority_root = context.AuthorityRoot ? context.AuthorityRoot.name : "UNAVAILABLE",
                avatar_root = context.AvatarRoot ? context.AvatarRoot.name : "UNAVAILABLE",
                torso = context.Torso ? context.Torso.name : "UNAVAILABLE",
                neck = context.Neck ? context.Neck.name : "UNAVAILABLE",
                head = context.Head ? context.Head.name : "UNAVAILABLE",
                camera_parent = context.HeadCameraMount && context.HeadCameraMount.parent ? context.HeadCameraMount.parent.name : "UNAVAILABLE",
                target_rigidbody = context.TargetBody ? context.TargetBody.name : "UNAVAILABLE",
                object_pose_writes_after_initialization = 0,
                object_external_forces = 0,
                attachment_or_joint_count = 0,
                assistance_ledger_entries = context.AssistanceLedger.Count,
                independent_render_timeline = false,
                single_state_drives_body_clothing_camera_truth = context.AvatarRoot && context.Head && context.HeadCameraMount && context.TargetBody,
                disclosure = "Kinematic embodiment commands and engine-observed poses are engineering control; the free target and contacts are PhysX-measured. No biological torque is claimed."
            };
            File.WriteAllText(
                Path.Combine(context.OutputRoot, "authority_receipt.json"),
                JsonUtility.ToJson(receipt, true)
            );
        }

        private static string RequiredEnvironment(string name)
        {
            string value = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(value)) throw new Exception(name + " is required");
            return value;
        }

        private static int RequiredInt(string name)
        {
            if (!int.TryParse(RequiredEnvironment(name), out int value)) throw new Exception(name + " must be an integer");
            return value;
        }

        [Serializable]
        private sealed class AuthorityReceipt
        {
            public string schema;
            public string episode_id;
            public string stage;
            public int physics_hz;
            public int render_hz;
            public int steps_per_render_frame;
            public int physics_steps;
            public int render_frames;
            public string authority_root;
            public string avatar_root;
            public string torso;
            public string neck;
            public string head;
            public string camera_parent;
            public string target_rigidbody;
            public int object_pose_writes_after_initialization;
            public int object_external_forces;
            public int attachment_or_joint_count;
            public int assistance_ledger_entries;
            public bool independent_render_timeline;
            public bool single_state_drives_body_clothing_camera_truth;
            public string disclosure;
        }
    }
}
#endif
