using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Leap;
using Leap.HandsModule;
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
        private const int SweepRampSteps = 96;
        private const int SweepHoldSteps = 24;
        private const int SweepCrossSteps = 240;
        private const int StepsPerDof = SweepRampSteps * 2 + SweepHoldSteps * 2 + SweepCrossSteps;
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
        private readonly Dictionary<int, Transform> commandSites = new Dictionary<int, Transform>();
        private readonly List<DofSpec> dofs = new List<DofSpec>();
        private readonly List<TraceRow> trace = new List<TraceRow>();
        private readonly List<ResetRow> resetLedger = new List<ResetRow>();
        private readonly Dictionary<string, Transform> visibleSites = new Dictionary<string, Transform>();
        private HandBinder[] visualBinders = new HandBinder[0];
        private Hand lastPostStepOutput;
        private string activeProfileSegment = "warmup";
        private bool activeSteadyState;
        private int currentActiveDof = -1;
        private float currentCommandAngle;
        private Hand lastCommand;

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
                active_right_hand_renderer_count = CountActiveRenderers(visibleRoot),
                active_left_hand_renderer_count = 0,
                right_hand_binder_count = visualBinders.Length,
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
            currentActiveDof = activeDof;
            int sweepStep = activeDof < 0 ? 0 : (step - WarmupSteps - SettleSteps) % StepsPerDof;
            float commandAngle = activeDof < 0 ? 0f : SweepProfile(sweepStep, out activeProfileSegment, out activeSteadyState);
            Hand command = MakeQualificationHand(step + 1, activeDof < 0 ? null : dofs[activeDof], commandAngle);
            currentCommandAngle = commandAngle;
            lastCommand = command;
            provider.EmitFixedFrame(command);
            Physics.Simulate(Dt);
            if (manager.RightHand != null)
            {
                Hand postStepOutput = BuildPostStepOutput(command);
                lastPostStepOutput = postStepOutput;
                SyncVisibleOutput(postStepOutput, command);
                CaptureTrace(command, postStepOutput, activeDof, commandAngle);
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
            diagnosticCamera.transform.position = new Vector3(.25f, .90f, -.58f);
            diagnosticCamera.transform.LookAt(new Vector3(0f, .86f, .24f));
            diagnosticCamera.fieldOfView = 35f;
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

                GameObject commandMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                commandMarker.name = $"diagnostic_{finger}_command_site";
                Collider commandCollider = commandMarker.GetComponent<Collider>();
                if (commandCollider != null) Destroy(commandCollider);
                commandMarker.transform.localScale = Vector3.one * .008f;
                commandMarker.GetComponent<Renderer>().material.color = Color.white;
                commandSites[finger] = commandMarker.transform;
            }
        }

        private void FindVisibleLandmarks()
        {
            GameObject visual = GameObject.Find("GenericHand_Arm_Visible");
            if (visual != null)
            {
                Transform left = visual.transform.Find("Left");
                if (left != null)
                {
                    foreach (Renderer renderer in left.GetComponentsInChildren<Renderer>(true)) renderer.enabled = false;
                    left.gameObject.SetActive(false);
                }
            }
            visibleRoot = visual == null ? null : visual.transform.Find("Right");
            if (visibleRoot == null && visual != null) visibleRoot = visual.transform;
            if (visibleRoot == null) return;
            foreach (Transform child in visibleRoot.GetComponentsInChildren<Transform>(true))
            {
                visibleSites[child.name] = child;
                if (child.name == "Hand_High_Arm" || child.name == "Elbow")
                    foreach (Renderer renderer in child.GetComponentsInChildren<Renderer>(true)) renderer.enabled = false;
            }
            visualBinders = visibleRoot.GetComponentsInChildren<HandBinder>(true);
            foreach (HandBinder binder in visualBinders)
            {
                binder.SetPositions = true;
                binder.SetModelScale = false;
            }
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
                dof_start_index_invariant = "dof_start_indices[ArticulationBody.index]",
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
                PerturbFinger(hand.fingers[sweep.finger], sweep.joint + 1, sweep.axis, angle);
            return hand;
        }

        private static float SweepProfile(int sweepStep, out string segment, out bool steadyState)
        {
            steadyState = false;
            if (sweepStep < SweepRampSteps)
            {
                segment = "ramp_down";
                return Mathf.Lerp(0f, -SweepAmplitudeRad, (sweepStep + 1f) / SweepRampSteps);
            }
            sweepStep -= SweepRampSteps;
            if (sweepStep < SweepHoldSteps)
            {
                segment = "hold_negative";
                steadyState = true;
                return -SweepAmplitudeRad;
            }
            sweepStep -= SweepHoldSteps;
            if (sweepStep < SweepCrossSteps)
            {
                segment = "ramp_up";
                return Mathf.Lerp(-SweepAmplitudeRad, SweepAmplitudeRad, (sweepStep + 1f) / SweepCrossSteps);
            }
            sweepStep -= SweepCrossSteps;
            if (sweepStep < SweepHoldSteps)
            {
                segment = "hold_positive";
                steadyState = true;
                return SweepAmplitudeRad;
            }
            segment = "return_neutral";
            return Mathf.Lerp(SweepAmplitudeRad, 0f, (sweepStep + 1f) / SweepRampSteps);
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

        private static void PerturbFinger(Finger finger, int boneIndex, int axisIndex, float angle)
        {
            if (axisIndex < 0 || axisIndex > 1) return;
            if (boneIndex < 1 || boneIndex >= finger.bones.Length) return;
            Bone pivotBone = finger.bones[boneIndex - 1];
            Vector3 pivot = pivotBone.PrevJoint;
            Vector3 axis = axisIndex == 1 ? pivotBone.Rotation * Vector3.up : pivotBone.Rotation * Vector3.right;
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

        private Hand BuildPostStepOutput(Hand command)
        {
            Hand outputHand = new Hand();
            outputHand.CopyFrom(command);
            ContactBone palm = manager.RightHand.palmBone;
            outputHand.PalmPosition = palm.transform.position;
            outputHand.StabilizedPalmPosition = palm.transform.position;
            outputHand.Rotation = palm.transform.rotation;
            for (int fingerIndex = 0; fingerIndex < outputHand.fingers.Length; fingerIndex++)
            {
                Finger outputFinger = outputHand.fingers[fingerIndex];
                for (int jointIndex = 0; jointIndex < 3; jointIndex++)
                {
                    CapsuleCollider capsule = manager.RightHand.GetBone(fingerIndex, jointIndex).GetComponent<CapsuleCollider>();
                    capsule.ToWorldSpaceCapsule(out Vector3 pointA, out Vector3 pointB, out float radius);
                    Bone outputBone = outputFinger.bones[jointIndex + 1];
                    if (jointIndex == 0)
                    {
                        outputFinger.bones[0].NextJoint = pointB;
                        outputBone.PrevJoint = pointB;
                    }
                    else outputBone.PrevJoint = pointB;
                    outputBone.NextJoint = pointA;
                    outputBone.Center = (outputBone.PrevJoint + outputBone.NextJoint) * .5f;
                    outputBone.Width = radius;
                    outputBone.Length = Vector3.Distance(outputBone.PrevJoint, outputBone.NextJoint);
                    outputBone.Direction = (outputBone.NextJoint - outputBone.PrevJoint).normalized;
                    outputBone.Rotation = manager.RightHand.GetBone(fingerIndex, jointIndex).transform.rotation;
                }
                CapsuleCollider distal = manager.RightHand.GetBone(fingerIndex, 2).GetComponent<CapsuleCollider>();
                distal.ToWorldSpaceCapsule(out Vector3 tipA, out Vector3 tipB, out float tipRadius);
                outputFinger.TipPosition = tipA + (tipA - tipB).normalized * tipRadius;
                outputFinger.Direction = (outputFinger.TipPosition - outputFinger.bones[0].PrevJoint).normalized;
            }
            return outputHand;
        }

        private void SyncVisibleOutput(Hand postStepOutput, Hand command)
        {
            if (visibleRoot == null || manager.RightHand == null) return;
            foreach (HandBinder binder in visualBinders)
            {
                binder.SetLeapHand(postStepOutput);
                binder.UpdateHand();
            }
            foreach (int finger in new[] { 0, 1, 2 })
            {
                diagnosticSites[finger].position = PostStepTip(manager.RightHand.GetBone(finger, 2));
                commandSites[finger].position = command.fingers[finger].TipPosition;
            }
        }

        private void SetVisible(string name, Transform physical)
        {
            if (!visibleSites.TryGetValue(name, out Transform visual)) return;
            visual.position = physical.position;
            visual.rotation = physical.rotation;
        }

        private static Vector3 PostStepTip(ContactBone bone)
        {
            CapsuleCollider capsule = bone.GetComponent<CapsuleCollider>();
            capsule.ToWorldSpaceCapsule(out Vector3 pointA, out Vector3 pointB, out float radius);
            return pointA + (pointA - pointB).normalized * radius;
        }

        private float VisibleRegistrationError(Hand postStepOutput)
        {
            if (visualBinders.Length == 0) return float.PositiveInfinity;
            float maximum = 0f;
            int measured = 0;
            string[][] landmarkMap = new string[][]
            {
                new[] { "R_thumb_a", "R_thumb_b" },
                new[] { "R_index_Proximal", "R_index_b", "R_index_c" },
                new[] { "R_middle_Proximal", "R_middle_b", "R_middle_c" },
                new[] { "R_ring_Proximal", "R_ring_b", "R_ring_c" },
                new[] { "R_pinky_Proximal", "R_pinky_b", "R_pinky_c" }
            };
            for (int finger = 0; finger < landmarkMap.Length; finger++)
            {
                for (int joint = 0; joint < landmarkMap[finger].Length; joint++)
                {
                    string name = landmarkMap[finger][joint];
                    if (string.IsNullOrEmpty(name) || !visibleSites.TryGetValue(name, out Transform visible)) continue;
                    int physicalJoint = Mathf.Min(joint, 2);
                    maximum = Mathf.Max(maximum, Vector3.Distance(visible.position, postStepOutput.fingers[finger].bones[physicalJoint + 1].PrevJoint));
                    measured++;
                }
            }
            return measured == 0 ? float.PositiveInfinity : maximum;
        }

        private static int CountActiveRenderers(Transform root)
        {
            if (root == null) return 0;
            return root.GetComponentsInChildren<Renderer>(true).Count(x => x.enabled && x.gameObject.activeInHierarchy);
        }

        private void CaptureTrace(Hand command, Hand physicsOutput, int activeDof, float commandAngle)
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
            float thumbError = Vector3.Distance(command.fingers[0].TipPosition, physicsOutput.fingers[0].TipPosition);
            float indexError = Vector3.Distance(command.fingers[1].TipPosition, physicsOutput.fingers[1].TipPosition);
            float middleError = Vector3.Distance(command.fingers[2].TipPosition, physicsOutput.fingers[2].TipPosition);
            float maxFingertipError = Mathf.Max(thumbError, Mathf.Max(indexError, middleError));
            float maxContactSiteFkError = 0f;
            for (int finger = 0; finger < 5; finger++)
                maxContactSiteFkError = Mathf.Max(maxContactSiteFkError, Vector3.Distance(physicsOutput.fingers[finger].TipPosition, PostStepTip(manager.RightHand.GetBone(finger, 2))));
            float maxVisibleError = VisibleRegistrationError(physicsOutput);
            bool activeReset = step >= WarmupSteps && (hand.Ghosted || !hand.Tracked);
            if (activeReset) resetLedger.Add(new ResetRow { physics_step = step, ghosted = hand.Ghosted, tracked = hand.Tracked });
            trace.Add(new TraceRow
            {
                physics_step = step,
                time_s = step * Dt,
                active_dof = activeDof,
                commanded_angle_rad = commandAngle,
                profile_segment = activeProfileSegment,
                steady_state = activeSteadyState,
                commanded_palm_m = command.PalmPosition,
                observed_palm_m = hand.palmBone.transform.position,
                tracked = hand.Tracked,
                ghosted = hand.Ghosted,
                max_fingertip_tracking_error_m = maxFingertipError,
                max_contact_site_fk_error_m = maxContactSiteFkError,
                thumb_tracking_error_m = thumbError,
                index_tracking_error_m = indexError,
                middle_tracking_error_m = middleError,
                max_visible_registration_error_m = maxVisibleError,
                visible_registration_status = visualBinders.Length > 0 ? "MEASURED_NAMED_LANDMARKS" : "NOT_MEASURED",
                output_frame_source = "post_step_articulation_capsule_fk",
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
            return body.index >= 0 && body.index < starts.Count ? starts[body.index] : -1;
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

        private void OnGUI()
        {
            if (Environment.GetEnvironmentVariable("REFERENCE_HAND_VISUAL") != "1") return;
            float thumb = lastCommand == null || lastPostStepOutput == null ? 0f : Vector3.Distance(lastCommand.fingers[0].TipPosition, lastPostStepOutput.fingers[0].TipPosition) * 1000f;
            float index = lastCommand == null || lastPostStepOutput == null ? 0f : Vector3.Distance(lastCommand.fingers[1].TipPosition, lastPostStepOutput.fingers[1].TipPosition) * 1000f;
            float middle = lastCommand == null || lastPostStepOutput == null ? 0f : Vector3.Distance(lastCommand.fingers[2].TipPosition, lastPostStepOutput.fingers[2].TipPosition) * 1000f;
            GUI.color = Color.white;
            GUI.Label(new Rect(24f, 22f, 900f, 28f), string.Format("DOF {0}  profile={1}  steady={2}  command={3:0.0} deg", currentActiveDof, activeProfileSegment, activeSteadyState, currentCommandAngle * Mathf.Rad2Deg));
            GUI.Label(new Rect(24f, 50f, 900f, 28f), string.Format("command minus post-step capsule FK: thumb={0:0.0} mm  index={1:0.0} mm  middle={2:0.0} mm", thumb, index, middle));
            GUI.Label(new Rect(24f, 78f, 900f, 28f), "white=commanded site  magenta/cyan/yellow=post-step contact site  visible route=right HandBinder");
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
                float baselineTarget = reference == null ? 0f : DriveForAxis(reference, spec.axis).target;
                float activeTargetMin = float.PositiveInfinity, activeTargetMax = float.NegativeInfinity, offAxisDelta = 0f;
                foreach (TraceRow row in trace.Where(x => x.active_dof == spec.index))
                {
                    ArticulationRow state = row.articulations.FirstOrDefault(x => x.body == spec.body);
                    if (state == null) continue;
                    DriveRow activeDrive = DriveForAxis(state, spec.axis);
                    activeTargetMin = Mathf.Min(activeTargetMin, activeDrive.target);
                    activeTargetMax = Mathf.Max(activeTargetMax, activeDrive.target);
                    for (int axis = 0; axis < 3; axis++)
                        if (axis != spec.axis) offAxisDelta = Mathf.Max(offAxisDelta, Mathf.Abs(DriveForAxis(state, axis).target - DriveForAxis(reference, axis).target));
                }
                float activeTargetRange = float.IsInfinity(activeTargetMin) ? 0f : activeTargetMax - activeTargetMin;
                bool activeTargetChanged = activeTargetRange >= 1f;
                bool offAxisBounded = offAxisDelta <= 5f;
                bool observed = !float.IsInfinity(min) && max - min > .02f;
                results.Add(new DofResult
                {
                    index = spec.index,
                    commanded_range_rad = SweepAmplitudeRad * 2f,
                    observed_range_rad = float.IsInfinity(min) ? 0f : max - min,
                    drive_target_range_deg = activeTargetRange,
                    off_axis_target_delta_deg = offAxisDelta,
                    controllable = controllable,
                    qualification = controllable ? "controllable" : "locked_ineligible",
                    active_drive_target_changed = activeTargetChanged,
                    off_axis_targets_bounded = offAxisBounded,
                    passed = controllable && observed && activeTargetChanged && offAxisBounded
                });
            }
            IEnumerable<TraceRow> qualifiedTrace = trace.Where(x => x.physics_step >= WarmupSteps);
            float maxTracking = !qualifiedTrace.Any() ? float.PositiveInfinity : qualifiedTrace.Max(x => x.max_fingertip_tracking_error_m);
            IEnumerable<TraceRow> steadyTrace = qualifiedTrace.Where(x => x.steady_state);
            float maxSteadyTracking = !steadyTrace.Any() ? float.PositiveInfinity : steadyTrace.Max(x => x.max_fingertip_tracking_error_m);
            float maxContactSiteFk = !qualifiedTrace.Any() ? float.PositiveInfinity : qualifiedTrace.Max(x => x.max_contact_site_fk_error_m);
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
                controllable_pass_count = results.Count(x => x.controllable && x.passed),
                dof_sweep_pass_count = results.Count(x => x.controllable && x.passed),
                dof_sweeps = results.ToArray(),
                max_fingertip_tracking_error_m = maxTracking,
                max_steady_state_fingertip_tracking_error_m = maxSteadyTracking,
                max_contact_site_fk_error_m = maxContactSiteFk,
                max_visible_registration_error_m = maxRegistration,
                max_commanded_palm_speed_m_s = maxCommandSpeed,
                active_phase_reset_count = resetLedger.Count,
                output_frame_source = "post_step_articulation_capsule_fk",
                visible_registration_status = visualBinders.Length > 0 ? "MEASURED_NAMED_LANDMARKS" : "NOT_MEASURED",
                stable_thumb_index_middle_tracking = maxSteadyTracking <= FingertipTrackingToleranceM,
                contact_site_fk_pass = maxContactSiteFk <= FingertipTrackingToleranceM,
                visible_collider_registration_pass = visualBinders.Length > 0 && maxRegistration <= VisibleRegistrationToleranceM,
                bounded_palm_speed_pass = maxCommandSpeed <= MaxCommandSpeedMPerS,
                passed = dofs.Count > 0 && results.Any(x => x.controllable) && results.Where(x => x.controllable).All(x => x.passed) && maxSteadyTracking <= FingertipTrackingToleranceM && maxContactSiteFk <= FingertipTrackingToleranceM && visualBinders.Length > 0 && maxRegistration <= VisibleRegistrationToleranceM && maxCommandSpeed <= MaxCommandSpeedMPerS && resetLedger.Count == 0
            };
        }

        private static float[] Values(ArticulationReducedSpace value)
        {
            float[] result = new float[value.dofCount];
            for (int i = 0; i < value.dofCount; i++) result[i] = value[i];
            return result;
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
            public int physics_hz, render_hz, steps_per_render_frame, grab_helper_components, grab_helper_object_runtime_instances, active_right_hand_renderer_count, active_left_hand_renderer_count, right_hand_binder_count;
            public bool provider_explicitly_assigned, physics_auto_simulation;
        }
        [Serializable] private sealed class DofManifest
        {
            public string schema;
            public int physical_body_count, dof_count;
            public int[] dof_start_indices;
            public string dof_start_index_invariant;
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
            public float time_s, commanded_angle_rad, max_fingertip_tracking_error_m, max_contact_site_fk_error_m, thumb_tracking_error_m, index_tracking_error_m, middle_tracking_error_m, max_visible_registration_error_m;
            public Vector3 commanded_palm_m, observed_palm_m, right_thumb_tip_m, right_index_tip_m, right_middle_tip_m, commanded_thumb_tip_m, commanded_index_tip_m, commanded_middle_tip_m, physics_output_thumb_tip_m, physics_output_index_tip_m, physics_output_middle_tip_m;
            public string profile_segment, visible_registration_status, output_frame_source;
            public bool tracked, ghosted, steady_state;
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
            public float commanded_range_rad, observed_range_rad, drive_target_range_deg, off_axis_target_delta_deg;
            public string qualification;
            public bool controllable, active_drive_target_changed, off_axis_targets_bounded, passed;
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
            public string output_frame_source, visible_registration_status;
            public int dof_count, controllable_dof_count, controllable_pass_count, dof_sweep_pass_count, active_phase_reset_count;
            public DofResult[] dof_sweeps;
            public float max_fingertip_tracking_error_m, max_steady_state_fingertip_tracking_error_m, max_contact_site_fk_error_m, max_visible_registration_error_m, max_commanded_palm_speed_m_s;
            public bool stable_thumb_index_middle_tracking, contact_site_fk_pass, visible_collider_registration_pass, bounded_palm_speed_pass, passed;
        }
    }
}
