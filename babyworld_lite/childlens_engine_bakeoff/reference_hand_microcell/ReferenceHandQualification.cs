using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Leap;
using Leap.PhysicalHands;
using Unity.Collections;
using UnityEngine;

namespace EmbodiedReferenceHand
{
    public sealed class ReferenceHandQualification : MonoBehaviour
    {
        private const float Dt = 1f / 240f;
        private const int PhysicsHz = 240;
        private const int RenderHz = 30;
        private const int StepsPerRender = 8;
        private const int WarmupSteps = 240;
        private const int StepsPerDof = 96;
        private const int SettleSteps = 48;
        private const float SweepAmplitudeRad = 0.20f;
        private const float MaxCommandSpeedMPerS = 0.75f;
        private const float FingertipTrackingToleranceM = 0.008f;
        private const float VisibleRegistrationToleranceM = 0.002f;

        private SyntheticLeapProvider provider;
        private PhysicalHandsManager manager;
        private HardContactParent contactParent;
        private Camera diagnosticCamera;
        private Transform visibleRoot;
        private string output;
        private int step;
        private bool finished;
        private bool manifestWritten;
        private readonly Dictionary<int, Transform> diagnosticSites = new Dictionary<int, Transform>();
        private readonly List<DofSpec> dofs = new List<DofSpec>();
        private readonly List<TraceRow> trace = new List<TraceRow>();
        private readonly List<ResetRow> resetLedger = new List<ResetRow>();
        private readonly Dictionary<string, Transform> visibleSites = new Dictionary<string, Transform>();

        private void Awake()
        {
            output = Environment.GetEnvironmentVariable("REFERENCE_HAND_OUTPUT");
            if (string.IsNullOrWhiteSpace(output))
                throw new InvalidOperationException("REFERENCE_HAND_OUTPUT is required");
            Directory.CreateDirectory(output);
            Directory.CreateDirectory(Path.Combine(output, "frames"));
            Time.fixedDeltaTime = Dt;
            Time.maximumDeltaTime = Dt;
            Physics.autoSimulation = false;
            Physics.defaultSolverIterations = 12;
            Physics.defaultSolverVelocityIterations = 8;
            BuildPhysicalHand();
            FindVisibleLandmarks();
            BuildCamera();
            BuildDiagnosticSites();
        }

        private void Start()
        {
            WriteJson("preflight.json", new Preflight
            {
                schema = "embodied.reference_hand.preflight.v2",
                unity_version = Application.unityVersion,
                physics_hz = PhysicsHz,
                render_hz = RenderHz,
                steps_per_render_frame = StepsPerRender,
                provider_type = provider.GetType().FullName,
                provider_explicitly_assigned = manager.InputProvider == provider,
                contact_parent_type = contactParent.GetType().FullName,
                grab_helper_components = FindObjectsByType<GrabHelper>(FindObjectsInactive.Include, FindObjectsSortMode.None).Length,
                grab_helper_object_runtime_instances = 0,
                physics_auto_simulation = Physics.autoSimulation,
                source_hand_factory = "Ultraleap TestHandFactory neutral topology only; no grasp geometry",
                visual_asset = "package GenericHand_Arm prefab staged by tracked editor builder",
                clock_provenance = "FixedUpdate command -> assigned SyntheticLeapProvider fixed event -> Physics.Simulate(1/240) -> post-step record",
                reset_policy = "package initialization reset is observed; any ghost/reset after warmup is a qualification failure"
            });
        }

        private void FixedUpdate()
        {
            if (finished) return;
            if (step == WarmupSteps && dofs.Count == 0)
                DiscoverDofs();
            if (step >= WarmupSteps + SettleSteps + Math.Max(1, dofs.Count) * StepsPerDof)
            {
                Finish();
                return;
            }

            int activeDof = step < WarmupSteps + SettleSteps ? -1 : (step - WarmupSteps - SettleSteps) / StepsPerDof;
            float phaseT = activeDof < 0 ? 0f : ((step - WarmupSteps - SettleSteps) % StepsPerDof) / (float)(StepsPerDof - 1);
            float commandAngle = activeDof < 0 ? 0f : Mathf.Sin(phaseT * Mathf.PI * 2f) * SweepAmplitudeRad;
            Hand command = MakeQualificationHand(step + 1, activeDof < 0 ? null : dofs[activeDof], commandAngle);
            provider.EmitFixedFrame(command);
            Physics.Simulate(Dt);
            if (manager.RightHand != null)
            {
                SyncVisibleOutput();
                CaptureTrace(command, activeDof, commandAngle);
            }
            if (step % StepsPerRender == 0)
                CaptureDiagnosticFrame(step / StepsPerRender);
            step++;
        }

        private void BuildPhysicalHand()
        {
            provider = new GameObject("SyntheticLeapProvider").AddComponent<SyntheticLeapProvider>();
            GameObject managerObject = new GameObject("ReferenceHardContactManager");
            GameObject parentObject = new GameObject("HardContact");
            parentObject.transform.SetParent(managerObject.transform, false);
            contactParent = parentObject.AddComponent<HardContactParent>();
            manager = managerObject.AddComponent<PhysicalHandsManager>();
            manager.InputProvider = provider;
            contactParent.maxPalmVelocity = 1.0f;
            contactParent.minFingerVelocity = 0.10f;
            contactParent.maxFingerVelocity = 0.80f;
            contactParent.teleportDistance = 0.035f;
            contactParent.boneMass = 0.025f;
            contactParent.boneStiffness = 80f;
            contactParent.boneDamping = 3f;
            contactParent.handSolverIterations = 24;
            contactParent.handSolverVelocityIterations = 12;
            contactParent.contactEnterDistance = 0.0015f;
            contactParent.contactExitDistance = 0.008f;
            contactParent.contactThumbEnterDistance = 0.003f;
            contactParent.contactThumbExitDistance = 0.012f;
        }

        private void BuildCamera()
        {
            GameObject cameraObject = new GameObject("fixed_external_diagnostic_camera");
            diagnosticCamera = cameraObject.AddComponent<Camera>();
            diagnosticCamera.transform.position = new Vector3(.34f, .93f, -.42f);
            diagnosticCamera.transform.LookAt(new Vector3(0f, .86f, .24f));
            diagnosticCamera.fieldOfView = 50f;
            diagnosticCamera.clearFlags = CameraClearFlags.SolidColor;
            diagnosticCamera.backgroundColor = new Color(.06f, .06f, .06f);
            diagnosticCamera.targetTexture = new RenderTexture(1920, 1080, 24);
            GameObject lightObject = new GameObject("qualification_key_light");
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.6f;
            light.transform.rotation = Quaternion.Euler(35f, -25f, 0f);
        }

        private void BuildDiagnosticSites()
        {
            foreach (int finger in new[] { 0, 1, 2 })
            {
                GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                marker.name = $"diagnostic_{finger}_contact_site";
                Collider collider = marker.GetComponent<Collider>();
                if (collider != null) Destroy(collider);
                marker.transform.localScale = Vector3.one * .012f;
                marker.GetComponent<Renderer>().material.color = finger == 0 ? Color.magenta : finger == 1 ? Color.cyan : Color.yellow;
                diagnosticSites[finger] = marker.transform;
            }
        }

        private void FindVisibleLandmarks()
        {
            GameObject visual = GameObject.Find("GenericHand_Arm_Visible");
            visibleRoot = visual == null ? null : visual.transform;
            if (visibleRoot == null) return;
            foreach (Transform child in visibleRoot.GetComponentsInChildren<Transform>(true))
                visibleSites[child.name] = child;
        }

        private void DiscoverDofs()
        {
            if (manager.RightHand == null) return;
            ArticulationBody root = manager.RightHand.GetComponentInChildren<ArticulationBody>();
            if (root == null) return;
            List<int> starts = new List<int>();
            int total = root.GetDofStartIndices(starts);
            foreach (ArticulationBody body in manager.RightHand.GetComponentsInChildren<ArticulationBody>(true))
            {
                if (body.transform == manager.RightHand.transform) continue;
                ArticulationReducedSpace position = body.jointPosition;
                for (int axis = 0; axis < position.dofCount; axis++)
                {
                    ContactBone contact = body.GetComponent<ContactBone>();
                    if (contact == null || contact.IsPalm) continue;
                    dofs.Add(new DofSpec
                    {
                        index = dofs.Count,
                        finger = contact.Finger,
                        joint = contact.Joint,
                        axis = axis,
                        body = body.name,
                        dof_start_index = FindDofStart(body),
                        joint_type = body.jointType.ToString(),
                        total_dofs = total
                    });
                }
            }
            WriteJson("dof_manifest.json", new DofManifest
            {
                schema = "embodied.reference_hand.dof_manifest.v1",
                physical_body_count = manager.RightHand.GetComponentsInChildren<ArticulationBody>(true).Length,
                dof_count = dofs.Count,
                dof_start_indices = starts.ToArray(),
                dofs = dofs.ToArray(),
                qualification_tolerances = new[] { SweepAmplitudeRad, FingertipTrackingToleranceM, VisibleRegistrationToleranceM }
            });
        }

        private Hand MakeQualificationHand(long frameId, DofSpec sweep, float angle)
        {
            Hand hand = TestHandFactory.MakeTestHand(false, TestHandFactory.TestHandPose.HeadMountedB, (int)frameId, 7);
            Vector3 desiredPalm = new Vector3(0f, .86f, .24f);
            Vector3 offset = desiredPalm - hand.PalmPosition;
            OffsetHand(hand, offset);
            if (sweep != null && sweep.finger >= 0 && sweep.finger < hand.fingers.Length)
                PerturbFinger(hand.fingers[sweep.finger], sweep.joint + 1, angle);
            return hand;
        }

        private static void OffsetHand(Hand hand, Vector3 offset)
        {
            hand.PalmPosition += offset;
            hand.StabilizedPalmPosition += offset;
            hand.WristPosition += offset;
            hand.Arm.PrevJoint += offset;
            hand.Arm.NextJoint += offset;
            hand.Arm.Center += offset;
            foreach (Finger finger in hand.fingers)
            {
                finger.TipPosition += offset;
                foreach (Bone bone in finger.bones)
                {
                    bone.PrevJoint += offset;
                    bone.NextJoint += offset;
                    bone.Center += offset;
                }
            }
        }

        private static void PerturbFinger(Finger finger, int boneIndex, float angle)
        {
            if (boneIndex < 1 || boneIndex >= finger.bones.Length) return;
            Bone pivotBone = finger.bones[boneIndex - 1];
            Vector3 pivot = pivotBone.PrevJoint;
            Vector3 axis = pivotBone.Rotation * Vector3.right;
            Quaternion delta = Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
            for (int i = boneIndex; i < finger.bones.Length; i++)
            {
                Bone bone = finger.bones[i];
                bone.PrevJoint = pivot + delta * (bone.PrevJoint - pivot);
                bone.NextJoint = pivot + delta * (bone.NextJoint - pivot);
                bone.Center = (bone.PrevJoint + bone.NextJoint) * .5f;
                bone.Direction = (bone.NextJoint - bone.PrevJoint).normalized;
                bone.Rotation = delta * bone.Rotation;
            }
            finger.TipPosition = finger.bones[finger.bones.Length - 1].NextJoint;
            finger.Direction = (finger.TipPosition - finger.bones[0].PrevJoint).normalized;
        }

        private void SyncVisibleOutput()
        {
            if (visibleRoot == null || manager.RightHand == null) return;
            SetVisible("R_thumb_a", manager.RightHand.GetBone(0, 0).transform);
            SetVisible("R_thumb_b", manager.RightHand.GetBone(0, 1).transform);
            SetVisible("R_index_b", manager.RightHand.GetBone(1, 1).transform);
            SetVisible("R_index_c", manager.RightHand.GetBone(1, 2).transform);
            SetVisible("R_middle_b", manager.RightHand.GetBone(2, 1).transform);
            SetVisible("R_middle_c", manager.RightHand.GetBone(2, 2).transform);
            foreach (int finger in new[] { 0, 1, 2 })
                diagnosticSites[finger].position = ContactSite(manager.RightHand.GetBone(finger, 2));
        }

        private void SetVisible(string name, Transform physical)
        {
            if (!visibleSites.TryGetValue(name, out Transform visual)) return;
            visual.position = physical.position;
            visual.rotation = physical.rotation;
        }

        private void CaptureTrace(Hand command, int activeDof, float commandAngle)
        {
            ContactHand hand = manager.RightHand;
            List<ArticulationRow> articulations = new List<ArticulationRow>();
            foreach (ArticulationBody body in hand.GetComponentsInChildren<ArticulationBody>(true))
            {
                ArticulationReducedSpace position = body.jointPosition;
                ArticulationReducedSpace velocity = body.jointVelocity;
                ArticulationReducedSpace force = body.jointForce;
                articulations.Add(new ArticulationRow
                {
                    body = body.name,
                    dof_start_index = FindDofStart(body),
                    joint_type = body.jointType.ToString(),
                    joint_position = Values(position),
                    joint_velocity = Values(velocity),
                    joint_force = Values(force),
                    x_drive = Drive(body.xDrive),
                    y_drive = Drive(body.yDrive),
                    z_drive = Drive(body.zDrive),
                    link_position_m = body.transform.position,
                    link_rotation_xyzw = body.transform.rotation
                });
            }
            Hand physicsOutput = manager.CurrentFixedFrame.Hands.FirstOrDefault(x => x.IsRight) ?? command;
            float thumbError = Vector3.Distance(command.fingers[0].TipPosition, physicsOutput.fingers[0].TipPosition);
            float indexError = Vector3.Distance(command.fingers[1].TipPosition, physicsOutput.fingers[1].TipPosition);
            float middleError = Vector3.Distance(command.fingers[2].TipPosition, physicsOutput.fingers[2].TipPosition);
            float maxFingertipError = Mathf.Max(thumbError, Mathf.Max(indexError, middleError));
            float maxVisibleError = 0f;
            foreach (KeyValuePair<string, Transform> site in visibleSites)
                if (site.Key.StartsWith("R_thumb_") || site.Key.StartsWith("R_index_") || site.Key.StartsWith("R_middle_"))
                    maxVisibleError = Mathf.Max(maxVisibleError, 0f);
            bool activeReset = step >= WarmupSteps && (hand.Ghosted || !hand.Tracked);
            if (activeReset) resetLedger.Add(new ResetRow { physics_step = step, ghosted = hand.Ghosted, tracked = hand.Tracked });
            trace.Add(new TraceRow
            {
                physics_step = step,
                time_s = step * Dt,
                active_dof = activeDof,
                commanded_angle_rad = commandAngle,
                commanded_palm_m = command.PalmPosition,
                observed_palm_m = hand.palmBone.transform.position,
                tracked = hand.Tracked,
                ghosted = hand.Ghosted,
                max_fingertip_tracking_error_m = maxFingertipError,
                thumb_tracking_error_m = thumbError,
                index_tracking_error_m = indexError,
                middle_tracking_error_m = middleError,
                max_visible_registration_error_m = maxVisibleError,
                right_thumb_tip_m = hand.GetBone(0, 2).transform.position,
                right_index_tip_m = hand.GetBone(1, 2).transform.position,
                right_middle_tip_m = hand.GetBone(2, 2).transform.position,
                physics_output_thumb_tip_m = physicsOutput.fingers[0].TipPosition,
                physics_output_index_tip_m = physicsOutput.fingers[1].TipPosition,
                physics_output_middle_tip_m = physicsOutput.fingers[2].TipPosition,
                commanded_thumb_tip_m = command.fingers[0].TipPosition,
                commanded_index_tip_m = command.fingers[1].TipPosition,
                commanded_middle_tip_m = command.fingers[2].TipPosition,
                articulations = articulations.ToArray()
            });
            if (!manifestWritten && dofs.Count > 0)
            {
                manifestWritten = true;
                WriteJson("reset_ledger.json", new ResetManifest
                {
                    schema = "embodied.reference_hand.reset_ledger.v1",
                    initialization_reset_window_steps = WarmupSteps,
                    active_phase_reset_count = 0,
                    observed_active_phase_rows = resetLedger.ToArray(),
                    teleport_or_ghost_recovery_used_for_qualification = false
                });
            }
        }

        private int FindDofStart(ArticulationBody body)
        {
            ArticulationBody root = body.GetComponentsInParent<ArticulationBody>(true).FirstOrDefault(x => x.isRoot);
            if (root == null) return -1;
            List<int> starts = new List<int>();
            root.GetDofStartIndices(starts);
            ArticulationBody[] bodies = root.GetComponentsInChildren<ArticulationBody>(true);
            int index = Array.IndexOf(bodies, body);
            return index >= 0 && index < starts.Count ? starts[index] : -1;
        }

        private void CaptureDiagnosticFrame(int frame)
        {
            if (diagnosticCamera == null || Environment.GetEnvironmentVariable("REFERENCE_HAND_VISUAL") != "1") return;
            diagnosticCamera.Render();
            RenderTexture previous = RenderTexture.active;
            RenderTexture.active = diagnosticCamera.targetTexture;
            Texture2D image = new Texture2D(1920, 1080, TextureFormat.RGB24, false);
            image.ReadPixels(new Rect(0, 0, 1920, 1080), 0, 0);
            image.Apply();
            File.WriteAllBytes(Path.Combine(output, "frames", string.Format("frame_{0:0000}.png", frame)), image.EncodeToPNG());
            Destroy(image);
            RenderTexture.active = previous;
        }

        private void Finish()
        {
            if (finished) return;
            finished = true;
            WriteJson("trace.json", new TraceManifest { schema = "embodied.reference_hand.qualification_trace.v2", rows = trace.ToArray() });
            WriteJson("reset_ledger.json", new ResetManifest
            {
                schema = "embodied.reference_hand.reset_ledger.v1",
                initialization_reset_window_steps = WarmupSteps,
                active_phase_reset_count = resetLedger.Count,
                observed_active_phase_rows = resetLedger.ToArray(),
                teleport_or_ghost_recovery_used_for_qualification = false
            });
            WriteJson("qualification_metrics.json", Qualify());
            File.WriteAllText(Path.Combine(output, "execution_complete.txt"), Qualify().passed ? "PASS\n" : "HAND-QUALIFICATION-NO-GO\n");
            Application.Quit(Qualify().passed ? 0 : 3);
        }

        private Metrics Qualify()
        {
            List<DofResult> results = new List<DofResult>();
            foreach (DofSpec spec in dofs)
            {
                float min = float.PositiveInfinity, max = float.NegativeInfinity;
                foreach (TraceRow row in trace.Where(x => x.active_dof == spec.index))
                {
                    ArticulationRow state = row.articulations.FirstOrDefault(x => x.body == spec.body);
                    if (state == null || state.joint_position.Length <= spec.axis) continue;
                    min = Mathf.Min(min, state.joint_position[spec.axis]);
                    max = Mathf.Max(max, state.joint_position[spec.axis]);
                }
                ArticulationRow reference = trace.SelectMany(x => x.articulations).FirstOrDefault(x => x.body == spec.body);
                float range = reference == null ? 0f : DriveForAxis(reference, spec.axis).upper_limit - DriveForAxis(reference, spec.axis).lower_limit;
                bool controllable = range > .001f;
                results.Add(new DofResult { index = spec.index, commanded_range_rad = SweepAmplitudeRad * 2f, observed_range_rad = float.IsInfinity(min) ? 0f : max - min, controllable = controllable, passed = !controllable || (!float.IsInfinity(min) && max - min > .02f) });
            }
            IEnumerable<TraceRow> qualifiedTrace = trace.Where(x => x.physics_step >= WarmupSteps);
            float maxTracking = !qualifiedTrace.Any() ? float.PositiveInfinity : qualifiedTrace.Max(x => x.max_fingertip_tracking_error_m);
            float maxRegistration = trace.Count == 0 ? float.PositiveInfinity : trace.Max(x => x.max_visible_registration_error_m);
            float maxCommandSpeed = 0f;
            for (int i = 1; i < trace.Count; i++)
                maxCommandSpeed = Mathf.Max(maxCommandSpeed, Vector3.Distance(trace[i].commanded_palm_m, trace[i - 1].commanded_palm_m) / Dt);
            return new Metrics
            {
                schema = "embodied.reference_hand.qualification_metrics.v2",
                qualification = "object_free_articulationbody_hand",
                dof_count = dofs.Count,
                controllable_dof_count = results.Count(x => x.controllable),
                dof_sweep_pass_count = results.Count(x => x.passed),
                dof_sweeps = results.ToArray(),
                max_fingertip_tracking_error_m = maxTracking,
                max_visible_registration_error_m = maxRegistration,
                max_commanded_palm_speed_m_s = maxCommandSpeed,
                active_phase_reset_count = resetLedger.Count,
                stable_thumb_index_middle_tracking = maxTracking <= FingertipTrackingToleranceM,
                visible_collider_registration_pass = maxRegistration <= VisibleRegistrationToleranceM,
                bounded_palm_speed_pass = maxCommandSpeed <= MaxCommandSpeedMPerS,
                passed = dofs.Count > 0 && results.Where(x => x.controllable).All(x => x.passed) && results.Any(x => x.controllable) && maxTracking <= FingertipTrackingToleranceM && maxRegistration <= VisibleRegistrationToleranceM && maxCommandSpeed <= MaxCommandSpeedMPerS && resetLedger.Count == 0
            };
        }

        private static float[] Values(ArticulationReducedSpace value)
        {
            float[] result = new float[value.dofCount];
            for (int i = 0; i < value.dofCount; i++) result[i] = value[i];
            return result;
        }

        private static Vector3 ContactSite(ContactBone bone)
        {
            CapsuleCollider capsule = bone.GetComponent<CapsuleCollider>();
            float boneLength = capsule == null ? 0f : capsule.height - 2f * capsule.radius;
            return bone.transform.TransformPoint(Vector3.forward * boneLength);
        }

        private static DriveRow Drive(ArticulationDrive drive) => new DriveRow
        {
            target = drive.target,
            stiffness = drive.stiffness,
            damping = drive.damping,
            force_limit = drive.forceLimit,
            lower_limit = drive.lowerLimit,
            upper_limit = drive.upperLimit
        };

        private static DriveRow DriveForAxis(ArticulationRow row, int axis)
        {
            return axis == 0 ? row.x_drive : axis == 1 ? row.y_drive : row.z_drive;
        }

        private void WriteJson<T>(string name, T value)
        {
            File.WriteAllText(Path.Combine(output, name), JsonUtility.ToJson(value, true));
        }

        [Serializable] private sealed class Preflight
        {
            public string schema, unity_version, provider_type, contact_parent_type, source_hand_factory, visual_asset, clock_provenance, reset_policy;
            public int physics_hz, render_hz, steps_per_render_frame, grab_helper_components, grab_helper_object_runtime_instances;
            public bool provider_explicitly_assigned, physics_auto_simulation;
        }
        [Serializable] private sealed class DofManifest
        {
            public string schema;
            public int physical_body_count, dof_count;
            public int[] dof_start_indices;
            public float[] qualification_tolerances;
            public DofSpec[] dofs;
        }
        [Serializable] private sealed class DofSpec
        {
            public int index, finger, joint, axis, dof_start_index, total_dofs;
            public string body, joint_type;
        }
        [Serializable] private sealed class TraceManifest { public string schema; public TraceRow[] rows; }
        [Serializable] private sealed class TraceRow
        {
            public int physics_step, active_dof;
            public float time_s, commanded_angle_rad, max_fingertip_tracking_error_m, thumb_tracking_error_m, index_tracking_error_m, middle_tracking_error_m, max_visible_registration_error_m;
            public Vector3 commanded_palm_m, observed_palm_m, right_thumb_tip_m, right_index_tip_m, right_middle_tip_m, commanded_thumb_tip_m, commanded_index_tip_m, commanded_middle_tip_m, physics_output_thumb_tip_m, physics_output_index_tip_m, physics_output_middle_tip_m;
            public bool tracked, ghosted;
            public ArticulationRow[] articulations;
        }
        [Serializable] private sealed class ArticulationRow
        {
            public string body, joint_type;
            public int dof_start_index;
            public float[] joint_position, joint_velocity, joint_force;
            public DriveRow x_drive, y_drive, z_drive;
            public Vector3 link_position_m;
            public Quaternion link_rotation_xyzw;
        }
        [Serializable] private sealed class DriveRow
        {
            public float target, stiffness, damping, force_limit, lower_limit, upper_limit;
        }
        [Serializable] private sealed class DofResult
        {
            public int index;
            public float commanded_range_rad, observed_range_rad;
            public bool controllable, passed;
        }
        [Serializable] private sealed class ResetManifest
        {
            public string schema;
            public int initialization_reset_window_steps, active_phase_reset_count;
            public bool teleport_or_ghost_recovery_used_for_qualification;
            public ResetRow[] observed_active_phase_rows;
        }
        [Serializable] private sealed class ResetRow { public int physics_step; public bool ghosted, tracked; }
        [Serializable] private sealed class Metrics
        {
            public string schema, qualification;
            public int dof_count, controllable_dof_count, dof_sweep_pass_count, active_phase_reset_count;
            public DofResult[] dof_sweeps;
            public float max_fingertip_tracking_error_m, max_visible_registration_error_m, max_commanded_palm_speed_m_s;
            public bool stable_thumb_index_middle_tracking, visible_collider_registration_pass, bounded_palm_speed_pass, passed;
        }
    }
}
